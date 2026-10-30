"""Two-candidate reranking restricted to label-free, close-score slates."""
import json
import time
import numpy as np
from arena_ood import ROOT,guard,write_json

OUT=ROOT/'competition/borderline_pair'
PROTOCOL=dict(version=1,source_parent='competition/final_v26',comparison_parent='competition/final_v27',
              geometry='competition/conditional_sphere',prior=10.,epoch=32,gap=.5,masses=[.25,.5,1.],
              selection='Require both origin AUC gains >1e-6, mean weighted gain >1e-6, no near-tie Top1 loss. Lock third origin after selection.')


def rerank(base,delta,mass,gap=.5):
    p=np.asarray(base,dtype=np.float64)
    assert p.ndim==2 and p.shape[1]==10 and p.shape==np.shape(delta)
    assert np.isfinite(p).all() and np.isfinite(delta).all() and ((p>=0)&(p<=1)).all()
    top=np.argsort(-p,axis=1,kind='stable')[:,:2];rows=np.arange(len(p))[:,None]
    pair=p[rows,top];logs=np.log(pair.clip(1e-8))
    active=(logs[:,0]-logs[:,1])<gap
    selected=np.flatnonzero(active);r=selected[:,None];k=top[selected]
    z=logs[selected]+mass*np.asarray(delta)[r,k]
    z-=z.max(1,keepdims=True);q=np.exp(z);q/=q.sum(1,keepdims=True)
    out=p.copy();out[r,k]=q*pair[selected].sum(1,keepdims=True)
    return out


def probe():
    from arena_conditional_sphere import metrics
    from arena_validation import importance_weights
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'selection.json').exists(),'Do not rerun or revise a completed probe.'
    pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    assert pointer['csv']==PROTOCOL['comparison_parent']+'/submission.csv'
    write_json(OUT/'protocol.json',PROTOCOL);write_json(OUT/'incumbent_pointer.json',pointer);before=guard()
    results={}
    for day in ['26','25']:
        raw=np.load(ROOT/PROTOCOL['source_parent']/('validation_'+day+'.npz'))
        previous=np.load(ROOT/PROTOCOL['comparison_parent']/('validation_'+day+'.npz'))
        source=np.load(ROOT/PROTOCOL['geometry']/('2025-01-'+day)/'prior10.0/validation_e32.npz')
        for k in ['event_indices','labels']:
            np.testing.assert_array_equal(raw[k],previous[k]);np.testing.assert_array_equal(raw[k],source[k])
        iw=importance_weights('2025-01-'+day,raw['event_indices'])[0]
        baseline=metrics(raw['labels'],previous['predictions'],raw['predictions'],iw);rows=[]
        for mass in PROTOCOL['masses']:
            p=rerank(raw['predictions'],source['correction'],mass,PROTOCOL['gap'])
            np.testing.assert_allclose(p.sum(1),raw['predictions'].sum(1),atol=1e-12,rtol=1e-12)
            row=dict(mass=mass,**metrics(raw['labels'],p,raw['predictions'],iw));rows.append(row)
            print(json.dumps(dict(stage='borderline_probe',day=day,**row,auc_delta=row['auc']-baseline['auc'])),flush=True)
        results[day]=dict(baseline=baseline,rows=rows)
        write_json(OUT/('result_'+day+'.json'),results[day])
    eligible=[]
    for a,b in zip(results['26']['rows'],results['25']['rows']):
        assert a['mass']==b['mass']
        ds=[r['auc']-results[d]['baseline']['auc'] for r,d in [(a,'26'),(b,'25')]]
        wd=np.mean([r['weighted_auc']-results[d]['baseline']['weighted_auc'] for r,d in [(a,'26'),(b,'25')]])
        top=[r['near_tie_top1']-results[d]['baseline']['near_tie_top1'] for r,d in [(a,'26'),(b,'25')]]
        if min(ds)>1e-6 and wd>1e-6 and min(top)>=0:
            eligible.append(dict(mass=a['mass'],gap=PROTOCOL['gap'],auc_26=a['auc'],auc_25=b['auc'],weighted_delta=float(wd),top1_deltas=top))
    selected=max(eligible,key=lambda a:a['weighted_delta']) if eligible else None
    write_json(OUT/'selection.json',dict(selected=selected,eligible=len(eligible),target_auc=.838))
    write_json(OUT/'state.json',dict(stage='qualified_needs_third_origin' if selected else 'completed_no_promotion',selected=selected,updated=time.time()))
    assert guard()==before
    lines=['# 只调整近分前两名的重排实验','',
           '先固定 V26 排序，只在前两名 log 概率差 < 0.5 时，用经过真实曝光条件训练的几何修正重排前两名；两名的概率总量和其余八名保持不变。',
           '以已经通过第三切分的 V27 为对照，预先固定三个幅度；还要求两个切分的近分 Top1 均不退步。','',
           '| 幅度 | Jan26 AUC | Jan25 AUC |','|---|---:|---:|']
    for a,b in zip(results['26']['rows'],results['25']['rows']):lines.append(f'| {a["mass"]} | {a["auc"]:.9f} | {b["auc"]:.9f} |')
    lines+=['',f'双切分合格：{len(eligible)}；不合格时保持 V27。',
            '目标 0.838 尚未达到；未使用官方评测标签。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(stage='borderline_completed',selected=selected)),flush=True)


if __name__=='__main__':probe()
