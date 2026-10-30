"""Constrained late reranking: change conditional preference, keep click mass.

Fixed label-free gates can focus a stronger preference expert on near ties or
short histories. Selection uses two chronological origins and still requires
an untouched third-origin check before promotion. No submission is written here.
"""
import argparse
import json
import numpy as np
from arena_ood import ROOT,write_json

OUT=ROOT/'competition/choice_gate_probe'


def event_gate(base,new,history,kind):
    lp=np.log(np.asarray(base,dtype=np.float64).clip(1e-8))
    top=np.sort(lp,axis=1)[:,-2:];gap=top[:,1]-top[:,0]
    if kind=='all':return np.ones(len(base))
    if kind=='near_tie':return (gap<.5).astype(float)
    if kind=='smooth_tie':return np.exp(-gap/.75)
    if kind=='short_history':return ((history!=0).sum(1)<10).astype(float)
    if kind in ['disagreement','confident_disagreement']:
        # Compare sets of tied maxima, so the gate cannot depend on candidate
        # position when a scorer is exactly uncertain between multiple items.
        a=base>=base.max(1,keepdims=True)-1e-12
        b=new>=new.max(1,keepdims=True)-1e-12
        disagreement=~(a&b).any(1)
        if kind=='disagreement':return disagreement.astype(float)
        ntop=np.sort(np.log(np.asarray(new,dtype=np.float64).clip(1e-8)),axis=1)[:,-2:]
        ngap=ntop[:,1]-ntop[:,0]
        return (disagreement&(gap<.75)&(ngap>gap+.25)).astype(float)
    raise ValueError(kind)


def mix(base,new,history,gate_kind,mass,kind='choice_log'):
    if mass==0:return base.copy()
    a=np.asarray(base,dtype=np.float64).clip(1e-8,1-1e-8)
    b=np.asarray(new,dtype=np.float64).clip(1e-8,1-1e-8)
    click=a.sum(1,keepdims=True)
    qa,qb=a/click,b/b.sum(1,keepdims=True)
    weight=(mass*event_gate(a,b,history,gate_kind))[:,None]
    if kind=='choice_probability':q=(1-weight)*qa+weight*qb
    elif kind=='choice_log':
        z=(1-weight)*np.log(qa)+weight*np.log(qb);z-=z.max(1,keepdims=True)
        q=np.exp(z);q/=q.sum(1,keepdims=True)
    else:raise ValueError(kind)
    return q*click


def run(names):
    from ctr_data import load_cache
    from sklearn.metrics import roc_auc_score
    from arena_validation import importance_weights,weighted_auc
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'incumbent_pointer.json').exists():
        write_json(OUT/'incumbent_pointer.json',json.loads((ROOT/'competition/CURRENT_BEST.json').read_text()))
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text());parent=(ROOT/pointer['csv']).parent
    _,raw=load_cache(ROOT/'data/cache_v1','train');groups={}
    baseline={}
    for day in ['26','25']:
        d=np.load(parent/('validation_'+day+'.npz'));iw,_=importance_weights('2025-01-'+day,d['event_indices'])
        groups[day]=(d,iw,raw['history'][d['event_indices']])
        baseline['auc_'+day]=float(roc_auc_score(d['labels'].ravel(),d['predictions'].ravel()))
        baseline['weighted_auc_'+day]=weighted_auc(d['labels'],d['predictions'],iw)
    baseline['score']=.5*(baseline['weighted_auc_26']+baseline['weighted_auc_25'])
    rows=[]
    for name in names:
        scored={day:np.load(ROOT/'competition/ood/runs'/name/('2025-01-'+day)/'validation.npz') for day in groups}
        for day,(d,_,_) in groups.items():
            for k in ['event_indices','labels']:np.testing.assert_array_equal(scored[day][k],d[k])
        for kind in ['choice_log','choice_probability']:
            for gate in ['all','near_tie','smooth_tie','short_history','disagreement','confident_disagreement']:
                for mass in [.1,.25,.5]:
                    r=dict(name=name,kind=kind,gate=gate,mass=mass)
                    for day,(d,iw,h) in groups.items():
                        p=mix(d['predictions'],scored[day]['predictions'],h,gate,mass,kind)
                        r['auc_'+day]=float(roc_auc_score(d['labels'].ravel(),p.ravel()))
                        r['weighted_auc_'+day]=weighted_auc(d['labels'],p,iw)
                    r['score']=.5*(r['weighted_auc_26']+r['weighted_auc_25']);rows.append(r)
    eligible=[r for r in rows if all(r[k]>baseline[k]+1e-6 for k in ['auc_26','auc_25','score'])]
    selected=max(eligible,key=lambda r:r['score']) if eligible else None
    write_json(OUT/'selection.json',dict(selected=selected,eligible=len(eligible),baseline=baseline,candidates=rows,
               scope='Two-origin preference-only probe, no CSV or incumbent change. Gates depend only on input history and frozen predictions. Needs locked third origin.'))
    print(json.dumps(dict(stage='completed',selected=selected,eligible=len(eligible))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);a=p.parse_args();run(a.names)
