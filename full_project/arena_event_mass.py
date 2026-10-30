"""Small chronological calibration of event click mass, with fixed choice.

Earlier out-of-time scores train a regularized event-level residual. Unlike
the preference reranker, this model preserves within-event candidate ratios.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import time
import numpy as np
import torch
from scipy.special import expit,logit
from arena_ood import ROOT,guard,write_json

OUT=ROOT/'competition/event_mass'
RECIPE=dict(version=1,ridges=[.01,.1,1.],masses=[.25,.5,1.],iterations=60,
            origins=2,bound=.35,seed=83903,objective='event_binary_cross_entropy',
            scalar_columns=[10,12,13,22,23,24,25,26,27,67,68,69,73,75])


def features(numeric,prob):
    p=np.asarray(prob,dtype='float64').clip(1e-8,1-1e-8)
    s=p.sum(1).clip(1e-6,1-1e-6);q=p/p.sum(1,keepdims=True)
    ordered=np.sort(q,axis=1);entropy=-(q*np.log(q)).sum(1)/np.log(10)
    x=np.asarray(numeric).reshape(len(p),10,-1)
    columns=[logit(s),entropy,ordered[:,-1],ordered[:,-2],np.log(ordered[:,-1]/ordered[:,-2]),
             (q*q).sum(1),logit(s)*entropy,entropy**2]
    columns += [x[:,0,i] for i in RECIPE['scalar_columns']]
    for k in [9,11,77]:columns += [x[:,:,k].mean(1),x[:,:,k].max(1),x[:,:,k].std(1)]
    f=np.stack(columns,1).astype('f');assert np.isfinite(f).all()
    return f


def apply(base,delta,mass):
    p=np.asarray(base,dtype=np.float64);s=p.sum(1,keepdims=True)
    assert np.all((s>0)&(s<1))
    target=expit(logit(s)+mass*np.asarray(delta)[:,None])
    return p*(target/s)


def inputs(frame,raw,cutoff=None):
    from arena_history_data import prepare
    from run import timestamp
    folder=prepare(frame);s=np.load(folder/'scores.npz');ix=s['event_indices']
    take=np.flatnonzero(raw['timestamps'][ix]<timestamp(cutoff)) if cutoff else np.arange(len(ix))
    matrix=np.load(folder/'features.npy',mmap_mode='r').reshape(len(ix),10,244)
    f=features(matrix,s['predictions'])
    return dict(features=f[take],prob=s['predictions'][take],event_indices=ix[take],labels=s['labels'][take])


def design(f,mean,scale,known):
    return np.concatenate([np.ones((len(f),1)),np.clip((f-mean)/scale,-6,6)*known],1).astype('f')


def correction(f,model):
    x=design(f,model['mean'],model['scale'],model['known'])
    return RECIPE['bound']*np.tanh(x@model['weights'])


def train(day,ridge,raw):
    from arena_two_stage_data import date_shift
    from run import timestamp
    path=OUT/day/('ridge'+str(ridge));path.mkdir(parents=True,exist_ok=True)
    if (path/'model.npz').exists():return dict(np.load(path/'model.npz'))
    chunks=[inputs(date_shift(day,-i),raw,day) for i in range(RECIPE['origins'],0,-1)]
    ix=np.concatenate([d['event_indices'] for d in chunks]);assert int(raw['timestamps'][ix].max())<timestamp(day)
    f=np.concatenate([d['features'] for d in chunks]);p=np.concatenate([d['prob'] for d in chunks]);labels=np.concatenate([d['labels'] for d in chunks])
    mean=f.mean(0);sd=f.std(0);known=(sd>1e-5).astype('f');scale=np.maximum(sd,1e-5)
    x=torch.tensor(design(f,mean,scale,known));offset=torch.tensor(logit(p.sum(1).clip(1e-6,1-1e-6)),dtype=torch.float32)
    y=torch.tensor(labels.sum(1)>0,dtype=torch.float32)
    _,inv,counts=np.unique(ix,return_inverse=True,return_counts=True)
    weights=torch.tensor(1/counts[inv],dtype=torch.float32);weights/=weights.mean()
    torch.manual_seed(RECIPE['seed']);coef=torch.nn.Parameter(torch.zeros(x.shape[1]))
    opt=torch.optim.LBFGS([coef],lr=.5,max_iter=RECIPE['iterations'],line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();delta=RECIPE['bound']*torch.tanh(x@coef)
        bce=(torch.nn.functional.binary_cross_entropy_with_logits(offset+delta,y,reduction='none')*weights).mean()
        loss=bce+.5*ridge*(delta.square()*weights).mean()+.0001*coef.square().mean()
        loss.backward();return loss
    opt.step(closure);model=dict(mean=mean,scale=scale,known=known,weights=coef.detach().numpy())
    np.savez(path/'model.npz',**model)
    write_json(path/'protocol.json',dict(day=day,ridge=ridge,events=len(ix),unique_events=len(counts),width=x.shape[1],
               frames=[date_shift(day,-i) for i in range(RECIPE['origins'],0,-1)],max_fit_timestamp=int(raw['timestamps'][ix].max()),
               constant_columns_disabled=np.flatnonzero(known==0).tolist(),policy='Earlier out-of-time first queries; all training targets before cutoff. Duplicate event views receive inverse-count weights.'))
    return model


def probe():
    from ctr_data import load_cache
    from arena_validation import importance_weights
    from arena_conditional_sphere import metrics
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    assert not (OUT/'selection.json').exists(),'Completed experiment is immutable.'
    pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text());write_json(OUT/'incumbent_pointer.json',pointer)
    write_json(OUT/'recipe.json',RECIPE);before=guard();_,raw=load_cache(ROOT/'data/cache_v1','train');results={}
    for day in ['26','25']:
        frame='2025-01-'+day;query=inputs(frame,raw)
        old=np.load((ROOT/pointer['csv']).parent/('validation_'+day+'.npz'))
        for k in ['event_indices','labels']:np.testing.assert_array_equal(old[k],query[k])
        iw=importance_weights(frame,old['event_indices'])[0];rows=[]
        baseline=metrics(old['labels'],old['predictions'],old['predictions'],iw)
        for ridge in RECIPE['ridges']:
            model=train(frame,ridge,raw);delta=correction(query['features'],model)
            np.savez_compressed(OUT/frame/('ridge'+str(ridge))/'validation.npz',event_indices=old['event_indices'],labels=old['labels'],correction=delta)
            for mass in RECIPE['masses']:
                p=apply(old['predictions'],delta,mass)
                row=dict(ridge=ridge,mass=mass,**metrics(old['labels'],p,old['predictions'],iw));rows.append(row)
                print(json.dumps(dict(stage='mass_probe',day=day,auc_delta=row['auc']-baseline['auc'],**row)),flush=True)
        results[day]=dict(baseline=baseline,records=rows);write_json(OUT/('result_'+day+'.json'),results[day])
    eligible=[]
    for a,b in zip(results['26']['records'],results['25']['records']):
        assert a['ridge']==b['ridge'] and a['mass']==b['mass']
        gain={d:r['auc']-results[d]['baseline']['auc'] for d,r in [('26',a),('25',b)]}
        wd=np.mean([r['weighted_auc']-results[d]['baseline']['weighted_auc'] for d,r in [('26',a),('25',b)]])
        if min(gain.values())>1e-6 and wd>1e-6:eligible.append(dict(ridge=a['ridge'],mass=a['mass'],auc_26=a['auc'],auc_25=b['auc'],deltas=gain,weighted_delta=float(wd)))
    selected=max(eligible,key=lambda r:r['weighted_delta']) if eligible else None
    write_json(OUT/'selection.json',dict(selected=selected,eligible=len(eligible),target_auc=.838))
    write_json(OUT/'state.json',dict(stage='qualified_needs_third_origin' if selected else 'completed_no_promotion',selected=selected,updated=time.time()))
    assert guard()==before
    lines=['# 第二轮只修正事件点击概率','',
           '31个时间样本外事件特征，32个线性系数，保留候选之间的概率比例；残差幅度限制在正负0.35，比较三档强度和三档使用幅度。',
           '早期模型预测只使用早于查询的训练样本。重复事件视图按倒数频次加权；训练期常数特征禁用。','',
           '| 正则 | 幅度 | Jan26 | Jan25 |','|---|---|---:|---:|']
    for a,b in zip(results['26']['records'],results['25']['records']):lines.append(f'| {a["ridge"]} | {a["mass"]} | {a["auc"]:.9f} | {b["auc"]:.9f} |')
    lines+=['',f'双切分合格 {len(eligible)} 组；目标 0.838 尚未达到。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(stage='mass_completed',selected=selected)),flush=True)


if __name__=='__main__':probe()
