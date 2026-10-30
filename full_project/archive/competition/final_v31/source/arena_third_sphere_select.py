"""Reapply six frozen consensus corrections to the current accepted parent.

Useful when another independent family finishes while representation fitting
runs. Never edit the original experiment or rely on its stale parent scores.
"""
import argparse
import json
import numpy as np
from arena_ood import ROOT,guard,write_json,sha
from arena_metric_geometry import preference_probabilities


def run(out):
    from arena_conditional_sphere import metrics
    from arena_validation import importance_weights
    source=ROOT/'competition/third_sphere'
    assert (source/'selection.json').exists(),'Finish the source campaign first.'
    assert not (out/'selection.json').exists(),'Use a fresh probe folder.'
    before=guard();out.mkdir(parents=True,exist_ok=True)
    pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text());write_json(out/'incumbent_pointer.json',pointer)
    protocol=json.loads((source/'protocol.json').read_text());write_json(out/'protocol.json',protocol)
    groups={};baselines={};results=[]
    for day in ['26','25']:
        p=np.load((ROOT/pointer['csv']).parent/('validation_'+day+'.npz'))
        iw=importance_weights('2025-01-'+day,p['event_indices'])[0]
        groups[day]=(p,iw);baselines[day]=metrics(p['labels'],p['predictions'],p['predictions'],iw)
    for fraction in protocol['third_masses']:
        ds={day:np.load(source/('correction_'+day+'_third'+str(fraction)+'.npz')) for day in groups}
        for day,(p,_) in groups.items():
            for k in ['event_indices','labels']:np.testing.assert_array_equal(p[k],ds[day][k])
        for mass in protocol['correction_masses']:
            row=dict(signal='trifused'+str(round(fraction*100))+'_delta',mass=mass)
            gains=[];weighted=[]
            for day,(p,iw) in groups.items():
                v=preference_probabilities(p['predictions'],ds[day]['correction'],mass)
                m=metrics(p['labels'],v,p['predictions'],iw);row['auc_'+day]=m['auc'];row['weighted_auc_'+day]=m['weighted_auc']
                row['near_tie_top1_'+day]=m['near_tie_top1'];gains.append(m['auc']-baselines[day]['auc'])
                weighted.append(m['weighted_auc']-baselines[day]['weighted_auc'])
            row['weighted_delta']=float(np.mean(weighted));row['eligible']=min(gains)>1e-6 and row['weighted_delta']>1e-6
            results.append(row)
    eligible=[r for r in results if r['eligible']];selected=max(eligible,key=lambda r:r['weighted_delta']) if eligible else None
    write_json(out/'selection.json',dict(selected=selected,eligible=len(eligible),records=results,baseline=baselines,
               source_protocol_sha256=sha(source/'protocol.json'),source_parent=json.loads((source/'incumbent_pointer.json').read_text()),current_parent=pointer,target_auc=.838))
    assert guard()==before;print(json.dumps(dict(stage='third_sphere_rebase',selected=selected)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(ROOT/a.out)
