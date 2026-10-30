"""Past-only, rotation-equivariant bilinear correction of OOF choice errors.

Fit a small matrix in each snapshot's own coordinate frame. This avoids a
global scoring network needing a stable basis across independent snapshots.
Only conditional item preference is modified; event click mass is preserved.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_ood import ROOT,sha,guard,write_json
from arena_oof_user_rerank import past_scores
from arena_metric_geometry import profiles,preference_probabilities

OUT=ROOT/'competition/metric_rerank'
GEOMETRY_MODE='current_snapshot'
CONFIGS=[dict(profiles=p,ridge=r) for p in [1,3] for r in [.001,.01,.1]]


class MetricCorrection(nn.Module):
    def __init__(self,n_profiles=3,dimension=8):
        super().__init__();self.matrix=nn.Parameter(torch.zeros(n_profiles,dimension,dimension))

    def forward(self,profile,candidates):
        q=torch.einsum('bkd,kde->be',profile[:,:len(self.matrix)],self.matrix)
        return torch.einsum('be,bie->bi',q,candidates)


def build_geometry(cutoff,raw,ix):
    from run import timestamp
    if GEOMETRY_MODE=='aligned_oof':
        from arena_aligned_repr import model
        assert np.all(raw['timestamps'][ix]>=timestamp(cutoff))
        rep=model(cutoff)
    else:
        from arena_spherical_repr import model
        rep=dict(np.load(model(cutoff)/'factors_e32.npz'))
    return profiles(raw,ix,rep)


def fitting_geometry(cutoff,raw,ix):
    if GEOMETRY_MODE!='aligned_oof':return build_geometry(cutoff,raw,ix)
    from datetime import datetime,timezone
    from arena_core import snapshot
    a=np.empty((len(ix),3,8),dtype=np.float32);b=np.empty((len(ix),10,8),dtype=np.float32)
    days=raw['timestamps'][ix]//86400
    for number in np.unique(days):
        day=datetime.fromtimestamp(int(number)*86400,timezone.utc).strftime('%Y-%m-%d')
        mask=days==number
        metadata=json.loads((snapshot(day)/'metadata.json').read_text())
        assert metadata['max_fit_timestamp']<int(raw['timestamps'][ix[mask]].min())
        a[mask],b[mask]=build_geometry(day,raw,ix[mask])
    return a,b


def fit(cutoff,index,raw):
    from run import timestamp
    folder=OUT/'fits'/cutoff/str(index);folder.mkdir(parents=True,exist_ok=True)
    config=CONFIGS[index]
    if (folder/'result.json').exists():
        assert json.loads((folder/'result.json').read_text())['config']==config
        return folder
    ix,p,y=past_scores(cutoff,raw)
    clicked=y.sum(1)>0;ix,p,y=ix[clicked],p[clicked],y[clicked]
    assert len(ix) and int(raw['timestamps'][ix].max())<timestamp(cutoff)
    profile,candidates=fitting_geometry(cutoff,raw,ix)
    profile=torch.from_numpy(profile);candidates=torch.from_numpy(candidates)
    logits=torch.from_numpy(np.log(p.clip(1e-8)).astype(np.float32))
    target=torch.from_numpy(y.argmax(1)).long()
    torch.set_num_threads(2);net=MetricCorrection(config['profiles'])
    # Deterministic full-batch L-BFGS has no held-out-day checkpoint selection.
    opt=torch.optim.LBFGS(net.parameters(),lr=1.,max_iter=35,max_eval=55,history_size=10,
                         tolerance_grad=2e-7,tolerance_change=1e-9,line_search_fn='strong_wolfe')
    records=[];tick=time.monotonic()
    def closure():
        opt.zero_grad();delta=net(profile,candidates)
        ce=F.cross_entropy(logits+delta,target)
        value=ce+.5*config['ridge']*net.matrix.square().sum()
        value.backward()
        records.append(dict(objective=float(value.detach()),choice_ce=float(ce.detach())))
        return value
    opt.step(closure)
    with torch.no_grad():
        ce=F.cross_entropy(logits+net(profile,candidates),target)
        before=F.cross_entropy(logits,target)
    np.savez(folder/'metric.npz',matrix=net.matrix.detach().numpy())
    write_json(folder/'result.json',dict(cutoff=cutoff,config=config,geometry_mode=GEOMETRY_MODE,training_clicked_events=len(ix),
               max_training_target_timestamp=int(raw['timestamps'][ix].max()),before_choice_ce=float(before),after_choice_ce=float(ce),
               objective_evaluations=len(records),seconds=time.monotonic()-tick,records=records,
               policy='Past daily time-OOF fine-ranker scores; '+('each geometry frame also strictly precedes its own fitting target.' if GEOMETRY_MODE=='aligned_oof' else 'current cutoff geometry fitted before held-out queries, but includes fitting clicks; self-influence risk.')+' No held-out labels in optimizer.'))
    print(json.dumps(dict(stage='metric_fit',cutoff=cutoff,config=config,before=float(before),after=float(ce),seconds=time.monotonic()-tick)),flush=True)
    return folder


def correction(raw,ix,cutoff,index):
    folder=fit(cutoff,index,raw);config=CONFIGS[index]
    net=MetricCorrection(config['profiles']);net.matrix.data.copy_(torch.from_numpy(np.load(folder/'metric.npz')['matrix']))
    parts=[]
    for start in range(0,len(ix),4096):
        p,v=build_geometry(cutoff,raw,ix[start:start+4096])
        with torch.no_grad():parts.append(net(torch.from_numpy(p),torch.from_numpy(v)).numpy())
    return np.concatenate(parts)


def probe(cutoff):
    from ctr_data import load_cache
    from arena_validation import importance_weights,weighted_auc
    from sklearn.metrics import roc_auc_score
    _,raw=load_cache(ROOT/'data/cache_v1','train')
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text());parent=(ROOT/pointer['csv']).parent
    d=np.load(parent/('validation_'+cutoff[-2:]+'.npz'));ix=d['event_indices'];labels=d['labels'];base=d['predictions']
    iw,_=importance_weights(cutoff,ix);rows=[]
    for i,config in enumerate(CONFIGS):
        delta=correction(raw,ix,cutoff,i)
        np.save(OUT/'fits'/cutoff/str(i)/'query_correction.npy',delta)
        for mass in [.25,.5,1.]:
            p=preference_probabilities(base,delta,mass)
            rows.append(dict(index=i,config=config,mass=mass,auc=float(roc_auc_score(labels.ravel(),p.ravel())),weighted_auc=weighted_auc(labels,p,iw)))
    result=dict(cutoff=cutoff,base=dict(auc=float(roc_auc_score(labels.ravel(),base.ravel())),weighted_auc=weighted_auc(labels,base,iw)),rows=rows)
    write_json(OUT/('probe_'+cutoff[-2:]+'.json'),result)
    return result


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'incumbent_pointer.json').exists():
        write_json(OUT/'incumbent_pointer.json',json.loads((ROOT/'competition/CURRENT_BEST.json').read_text()))
        write_json(OUT/'incumbent_digest.json',guard())
    assert guard()==json.loads((OUT/'incumbent_digest.json').read_text())
    a,b=probe('2025-01-26'),probe('2025-01-25');rows=[]
    for x,y in zip(a['rows'],b['rows']):
        assert x['index']==y['index'] and x['mass']==y['mass']
        r=dict(index=x['index'],config=x['config'],mass=x['mass'])
        for day,v in [('26',x),('25',y)]:r.update({'auc_'+day:v['auc'],'weighted_auc_'+day:v['weighted_auc']})
        r['score']=.5*(x['weighted_auc']+y['weighted_auc']);rows.append(r)
    eligible=[r for r in rows if r['auc_26']>a['base']['auc']+1e-6 and r['auc_25']>b['base']['auc']+1e-6 and
              r['score']>.5*(a['base']['weighted_auc']+b['base']['weighted_auc'])+1e-6]
    selected=max(eligible,key=lambda r:r['score']) if eligible else None
    write_json(OUT/'selection.json',dict(selected=selected,eligible=len(eligible),candidates=rows,
               scope='Dual-origin probe. Any selected recipe still needs locked third origin, exact CSV validation and frozen-source raw replay.'))
    assert guard()==json.loads((OUT/'incumbent_digest.json').read_text())
    print(json.dumps(dict(stage='completed',selected=selected,eligible=len(eligible))),flush=True)


if __name__=='__main__':run()
