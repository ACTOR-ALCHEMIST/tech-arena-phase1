"""Read-only probe: replace stale preference evidence with refined geometry.

Add a small cosine correction to conditional candidate logits while preserving
the incumbent's event click probability. Old-cosine additions are controls.
"""
import argparse
import json
import numpy as np
from arena_ood import ROOT,write_json
from arena_metric_geometry import preference_probabilities,cosine

OUT=ROOT/'competition/latent_delta_probe'
INCLUDE_ALT=False
MASSES=[.1,.25,.5,1.]


def signals(raw,ix,cutoff):
    from arena_spherical_repr import model as old_model
    from arena_spherical_long import model as long_model
    from arena_fused_sphere import model as fused_model
    from run import timestamp
    assert np.all(raw['timestamps'][ix]>=timestamp(cutoff))
    old=cosine(dict(np.load(old_model(cutoff)/'factors_e32.npz')),raw,ix)
    new=cosine(dict(np.load(long_model(cutoff)/'factors_e128.npz')),raw,ix)
    fused=fused_model(cutoff)
    f50=cosine(dict(np.load(fused/'w50.npz')),raw,ix)
    f75=cosine(dict(np.load(fused/'w75.npz')),raw,ix)
    result=dict(old32=old,long128=new,fused50=f50,fused75=f75,
                long_delta=new-old,fused50_delta=f50-old,fused75_delta=f75-old)
    if INCLUDE_ALT:
        from arena_spherical_alt import model as alt_model
        from arena_fused_sphere import fuse
        from arena_core import snapshot
        a=dict(np.load(long_model(cutoff)/'factors_e128.npz'))
        b=dict(np.load(alt_model(cutoff)/'factors_e128.npz'))
        alt=cosine(b,raw,ix);result['alt128']=alt;result['alt_delta']=alt-old
        stats=dict(np.load(snapshot(cutoff)/'stats.npz'))
        known=np.linalg.norm(a['users'],axis=1)>0
        for mass in [.25,.5,.75]:
            tag='unitfused'+str(round(100*mass))
            value=cosine(fuse(a,b,known,stats['pop'],mass),raw,ix)
            result[tag]=value;result[tag+'_delta']=value-old
    return result


def run():
    from ctr_data import load_cache
    from sklearn.metrics import roc_auc_score
    from arena_validation import importance_weights,weighted_auc
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'incumbent_pointer.json').exists():
        write_json(OUT/'incumbent_pointer.json',json.loads((ROOT/'competition/CURRENT_BEST.json').read_text()))
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text());parent=(ROOT/pointer['csv']).parent
    _,raw=load_cache(ROOT/'data/cache_v1','train');groups={};base={}
    for day in ['26','25']:
        d=np.load(parent/('validation_'+day+'.npz'));s=signals(raw,d['event_indices'],'2025-01-'+day)
        np.savez_compressed(OUT/('signals_'+day+'.npz'),**s)
        iw,_=importance_weights('2025-01-'+day,d['event_indices']);groups[day]=(d,s,iw)
        base['auc_'+day]=float(roc_auc_score(d['labels'].ravel(),d['predictions'].ravel()))
        base['weighted_auc_'+day]=weighted_auc(d['labels'],d['predictions'],iw)
    base['score']=.5*(base['weighted_auc_26']+base['weighted_auc_25']);rows=[]
    for name in groups['26'][1]:
        for mass in MASSES:
            row=dict(signal=name,mass=mass)
            for day,(d,s,iw) in groups.items():
                p=preference_probabilities(d['predictions'],s[name],mass)
                row['auc_'+day]=float(roc_auc_score(d['labels'].ravel(),p.ravel()))
                row['weighted_auc_'+day]=weighted_auc(d['labels'],p,iw)
            row['score']=.5*(row['weighted_auc_26']+row['weighted_auc_25']);rows.append(row)
    eligible=[r for r in rows if all(r[k]>base[k]+1e-6 for k in ['auc_26','auc_25','score'])]
    selected=max(eligible,key=lambda r:r['score']) if eligible else None
    write_json(OUT/'selection.json',dict(selected=selected,eligible=len(eligible),baseline=base,recipes=rows,include_alt=INCLUDE_ALT,masses=MASSES,
               scope='Read-only two-origin probe; old-cosine controls, fixed small grid, no CSV or incumbent change. Needs third origin and raw replay.'))
    print(json.dumps(dict(stage='completed',selected=selected,eligible=len(eligible))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--include-alt',action='store_true');p.add_argument('--extend',action='store_true');a=p.parse_args()
    if a.out:OUT=ROOT/a.out
    INCLUDE_ALT=a.include_alt
    if a.extend:MASSES=[.1,.25,.5,1.,1.5,2.]
    run()
