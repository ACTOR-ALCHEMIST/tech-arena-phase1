"""CPU probe: shrink user-specific preference residuals against OOF logits.

Unlike the earlier Laplace probe, offsets are honest fine-ranker predictions.
Only past OOF targets fit the user correction; current query targets are scored
afterwards. This probe never writes CURRENT_BEST or a submission CSV.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import time
import numpy as np
from numba import set_num_threads
from arena_ood import ROOT,write_json,sha
from arena_two_stage_data import OUT as DATA,date_shift
from arena_laplace_solver import solve_users

OUT=ROOT/'competition/oof_user_rerank'
CONFIGS=[dict(latent_precision=p,bias_precision=b) for p,b in [(20.,100.),(20.,1e6),(100.,100.),(100.,1e6)]]


def past_scores(cutoff,raw,days=2):
    from run import timestamp
    parts=[np.load(DATA/'inputs'/('oof_'+date_shift(cutoff,-d))/'scores.npz') for d in range(days,0,-1)]
    ix=np.concatenate([p['event_indices'] for p in parts]);p=np.concatenate([x['predictions'] for x in parts])
    y=np.concatenate([x['labels'] for x in parts])
    assert len(np.unique(ix))==len(ix)
    assert int(raw['timestamps'][ix].max())<timestamp(cutoff)
    np.testing.assert_array_equal(y,raw['labels'][ix])
    for d,part in zip(range(days,0,-1),parts):
        assert int(raw['timestamps'][part['event_indices']].min())>=timestamp(date_shift(cutoff,-d))
    return ix,p,y


def probabilities(base,raw,indices,delta,vectors,mode,mass):
    p=np.asarray(base,dtype=np.float64).clip(1e-8,1-1e-8)
    correction=np.einsum('bik,bk->bi',vectors[raw['items'][indices]],delta[raw['users'][indices],:-1])
    correction+=delta[raw['users'][indices],-1,None]
    if mode=='choice':
        z=np.log(p)+mass*correction;z-=z.max(1,keepdims=True)
        q=np.exp(z);return q/q.sum(1,keepdims=True)*p.sum(1,keepdims=True)
    outside=(1-p.sum(1,keepdims=True)).clip(1e-8)
    z=np.concatenate([np.log(p)+mass*correction,np.log(outside)],1);z-=z.max(1,keepdims=True)
    q=np.exp(z);return (q/q.sum(1,keepdims=True))[:,:10]


def probe(cutoff,base_folder):
    from ctr_data import load_cache
    from arena_spherical_repr import model as sphere_model
    from arena_validation import importance_weights,weighted_auc
    from sklearn.metrics import roc_auc_score
    meta,raw=load_cache(ROOT/'data/cache_v1','train');set_num_threads(2)
    ix,p,y=past_scores(cutoff,raw);order=np.argsort(raw['users'][ix],kind='stable')
    ix,p,y=ix[order],p[order],y[order]
    count=np.bincount(raw['users'][ix],minlength=meta['n_users'])
    indptr=np.r_[0,np.cumsum(count)].astype(np.int64)
    vectors=np.load(sphere_model(cutoff)/'factors_e32.npz')['items'].astype(np.float64)
    vectors/=np.linalg.norm(vectors,axis=1,keepdims=True).clip(1e-8)
    outside=(1-p.sum(1,keepdims=True)).clip(1e-8)
    offsets=np.log(np.concatenate([p,outside],1).clip(1e-8)).astype(np.float64)
    target=np.where(y.sum(1)>0,y.argmax(1),10).astype(np.int64)
    valid=np.load(base_folder/('validation_'+cutoff[-2:]+'.npz'))
    vi=valid['event_indices'];baseline=valid['predictions'];labels=valid['labels']
    assert not np.intersect1d(ix,vi).size
    iw,_=importance_weights(cutoff,vi)
    base=dict(auc=float(roc_auc_score(labels.ravel(),baseline.ravel())),weighted_auc=weighted_auc(labels,baseline,iw))
    rows=[]
    for n,config in enumerate(CONFIGS):
        folder=OUT/cutoff/str(n);folder.mkdir(parents=True,exist_ok=True)
        precision=np.full((meta['n_users'],9),config['latent_precision'],np.float64);precision[:,-1]=config['bias_precision']
        tick=time.monotonic()
        if (folder/'posterior.npz').exists():delta=np.load(folder/'posterior.npz')['delta']
        else:
            delta,cov,loss,grad,steps=solve_users(indptr,np.array(raw['items'][ix]),offsets,target,vectors,1.,precision,12)
            np.savez_compressed(folder/'posterior.npz',delta=delta)
            write_json(folder/'fit.json',dict(cutoff=cutoff,config=config,training_events=len(ix),
                       maximum_fit_timestamp=int(raw['timestamps'][ix].max()),max_gradient=float(grad.max()),seconds=time.monotonic()-tick,
                       fitting_policy='Per-user posterior from strictly past time-OOF fine-ranker offsets; current query targets never used.'))
        for mode in ['choice','joint']:
            for mass in [.25,.5,1.]:
                new=probabilities(baseline,raw,vi,delta,vectors,mode,mass)
                rows.append(dict(config_index=n,mode=mode,mass=mass,auc=float(roc_auc_score(labels.ravel(),new.ravel())),
                                 weighted_auc=weighted_auc(labels,new,iw)))
        print(json.dumps(dict(stage='oof_user_posterior',cutoff=cutoff,config=config,seconds=time.monotonic()-tick)),flush=True)
    write_json(OUT/cutoff/'probe.json',dict(base=base,rows=rows,baseline_folder=str(base_folder.relative_to(ROOT))))
    return base,rows


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'incumbent_pointer.json').exists():
        write_json(OUT/'incumbent_pointer.json',json.loads((ROOT/'competition/CURRENT_BEST.json').read_text()))
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text());base_folder=(ROOT/pointer['csv']).parent
    groups={d:probe('2025-01-'+d,base_folder) for d in ['26','25']}
    rows=[]
    for a,b in zip(groups['26'][1],groups['25'][1]):
        assert (a['config_index'],a['mode'],a['mass'])==(b['config_index'],b['mode'],b['mass'])
        r={k:a[k] for k in ['config_index','mode','mass']}
        for day,value in [('26',a),('25',b)]:r.update({k+'_'+day:value[k] for k in ['auc','weighted_auc']})
        r['score']=.5*(a['weighted_auc']+b['weighted_auc']);rows.append(r)
    eligible=[r for r in rows if all(r['auc_'+d]>groups[d][0]['auc']+1e-6 for d in groups) and
              r['score']>.5*sum(g[0]['weighted_auc'] for g in groups.values())+1e-6]
    chosen=max(eligible,key=lambda r:r['score']) if eligible else None
    write_json(OUT/'selection.json',dict(selected=chosen,eligible=len(eligible),rows=rows,baseline=pointer,
               scope='Exploratory CPU probe only. No CSV or incumbent change; needs locked third origin and full replay before any promotion.'))
    print(json.dumps(dict(stage='completed',selected=chosen,eligible=len(eligible))),flush=True)


if __name__=='__main__':run()
