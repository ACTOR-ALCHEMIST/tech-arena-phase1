"""Lock third-origin evidence, refit the accepted slate geometry and replay CSV."""
import json
import shutil
import subprocess
import sys
import time
import numpy as np
from arena_ood import ROOT,guard,write_json,sha
from arena_conditional_sphere import OUT,run,unit,metrics
from arena_metric_geometry import cosine,preference_probabilities

FINAL=ROOT/'competition/final_v27'


def export(selected,pointer):
    from ctr_data import load_cache
    from arena_latent_delta_export import component
    from arena_metric_infer import read_csv
    from arena_finalize import export_csv
    from arena_core import snapshot
    from run import timestamp
    before=guard();assert not (FINAL/'manifest.json').exists(),'Use a new release folder.'
    run('2025-01-27',selected)
    folder=OUT/'2025-01-27';result=json.loads((folder/'result.json').read_text())
    temperature=result['calibration']['coefficients'][0]
    old=component('2025-01-27','unitfused25_delta')[1]
    old={k:unit(old[k]) for k in ['users','items']}
    path=folder/('prior'+str(selected['prior']))/('geometry_e'+str(selected['epoch'])+'.npz')
    new=dict(np.load(path));FINAL.mkdir(parents=True,exist_ok=True)
    for name,rep in [('old',old),('new',new)]:np.savez(FINAL/(name+'_geometry.npz'),**rep)
    shutil.copy2(ROOT/pointer['csv'],FINAL/'frozen_incumbent.csv')
    _,raw=load_cache(ROOT/'data/cache_v1','evaluation')
    base=read_csv(FINAL/'frozen_incumbent.csv',raw['rowids'].ravel());pieces=[]
    for start in range(0,len(raw['users']),4096):
        ix=np.arange(start,min(start+4096,len(raw['users'])))
        pieces.append(cosine(new,raw,ix)-cosine(old,raw,ix))
    delta=np.concatenate(pieces);effective_mass=selected['mass']*temperature
    p=preference_probabilities(base,delta,effective_mass)
    trained=np.load(path.parent/'evaluation.npz')['correction']
    np.testing.assert_allclose(delta*temperature,trained,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(p.sum(1),base.sum(1),atol=1e-12,rtol=1e-12)
    np.save(FINAL/'predictions.npy',p);np.save(FINAL/'correction.npy',delta);export_csv(FINAL/'submission.csv',raw,p)
    for d in ['26','25','24']:
        shutil.copy2(OUT/('selected_validation_'+d+'.npz'),FINAL/('validation_'+d+'.npz'))
        shutil.copy2(OUT/('audit_'+d+'.json'),FINAL/('audit_'+d+'.json'))
    shutil.copy2(OUT/'selection.json',FINAL/'selection.json');shutil.copy2(folder/'result.json',FINAL/'refit.json')
    metadata=json.loads((snapshot('2025-01-27')/'metadata.json').read_text())
    assert result['max_fit_timestamp']<timestamp('2025-01-27') and metadata['max_fit_timestamp']<timestamp('2025-01-27')
    write_json(FINAL/'representation_provenance.json',dict(snapshot=metadata,training=result,source_geometry_sha256=sha(path),
               policy='Earlier geometry calibrates seven global coefficients; fixed-temperature, support-normalized spherical MAP uses actual clicked slates strictly before cutoff. Query labels unavailable.'))
    source=FINAL/'source';source.mkdir(exist_ok=True)
    for f in list(ROOT.glob('arena_*.py'))+[ROOT/n for n in ['run.py','ctr_data.py','ctr_model.py','test_arena_conditional_sphere.py']]:
        shutil.copy2(f,source/f.name)
    manifest=dict(method='conditional_clicked_slate_spherical_map',selection=dict(mass=float(effective_mass)),
                  locked_slate_recipe=selected,calibrated_temperature=float(temperature),
                  formula='p_new = sum(p_parent) * softmax(log(p_parent) + mass * temperature * (cos_new - cos_initial))',
                  subtract_old=True,cutoff_timestamp=timestamp('2025-01-27'),target_auc=.838,
                  official_evaluation_labels_available=False,previous_best=pointer,csv=str((FINAL/'submission.csv').relative_to(ROOT)),
                  sha256=sha(FINAL/'submission.csv'),rows=int(p.size),incumbent_digest=before,
                  **{'validation_'+d:dict(auc=json.loads((OUT/('audit_'+d+'.json')).read_text())['new_auc']) for d in ['26','25','24']},
                  artifact_sha256={str(f.relative_to(FINAL)):sha(f) for f in FINAL.rglob('*') if f.is_file() and f.name not in ['manifest.json','acceptance.json']},
                  raw_input_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'data/cache_v1/evaluation').glob('*.npy')})
    write_json(FINAL/'manifest.json',manifest);assert guard()==before
    subprocess.run([sys.executable,'-u',str(source/'arena_latent_delta_infer.py'),'--final',str(FINAL)],check=True,cwd=ROOT)
    assert guard()==before and json.loads((FINAL/'acceptance.json').read_text())['passed']
    write_json(ROOT/'competition/CURRENT_BEST.json',dict(csv=manifest['csv'],sha256=manifest['sha256'],
               manifest=str((FINAL/'manifest.json').relative_to(ROOT)),acceptance=str((FINAL/'acceptance.json').relative_to(ROOT)),
               previous_best_csv=pointer['csv'],**{'validation_'+d:manifest['validation_'+d] for d in ['26','25','24']},
               scope='Best validated local recipe; official evaluation labels unavailable.'))
    subprocess.run([sys.executable,'-u',str(source/'arena_latent_delta_infer.py'),'--final',str(FINAL),'--check-only'],check=True,cwd=ROOT)


def finish():
    from arena_push084_audit import compare
    from arena_validation import importance_weights
    selection=json.loads((OUT/'selection.json').read_text());selected=selection['selected']
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text())
    assert pointer==json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    if selected is None:return
    before=guard();run('2025-01-24',selected)
    parent=(ROOT/pointer['csv']).parent;audits={};edges={}
    for day in ['26','25','24']:
        baseline=np.load(parent/('validation_'+day+'.npz'));folder=OUT/('2025-01-'+day)/('prior'+str(selected['prior']))
        scored=np.load(folder/('validation_e'+str(selected['epoch'])+'.npz'))
        for k in ['event_indices','labels']:np.testing.assert_array_equal(baseline[k],scored[k])
        p=preference_probabilities(baseline['predictions'],scored['correction'],selected['mass'])
        new=dict(event_indices=scored['event_indices'],labels=scored['labels'],predictions=p)
        np.savez_compressed(OUT/('selected_validation_'+day+'.npz'),**new)
        audits[day]=compare(baseline,new,'2025-01-'+day)
        audits[day]['locked_selection_sha256']=sha(OUT/'selection.json')
        write_json(OUT/('audit_'+day+'.json'),audits[day])
        iw=importance_weights('2025-01-'+day,scored['event_indices'])[0]
        a=metrics(scored['labels'],baseline['predictions'],baseline['predictions'],iw)
        b=metrics(scored['labels'],p,baseline['predictions'],iw)
        edges[day]=dict(before=a,after=b)
        if day in ['26','25']:assert abs(audits[day]['new_auc']-selected['auc_'+day])<1e-12
    third=audits['24'];anchor=json.loads((ROOT/'competition/final_v6/audit_24.json').read_text())
    passed=(third['delta']>=-.0001 and third['cohorts']['weekday']['delta']>=-.0001 and
            third['new_auc']>=anchor['new_auc']-.0001 and third['cohorts']['weekday']['new_auc']>=anchor['cohorts']['weekday']['new_auc']-.0001)
    write_json(OUT/'promotion.json',dict(passed=passed,audit_deltas={d:a['delta'] for d,a in audits.items()},near_tie_edges=edges,target_auc=.838))
    assert guard()==before
    if passed:export(selected,pointer)
    current=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    lines=['# 真实候选集合中的点击偏好修正','',
           '第一轮保留 V26 概率；第二轮以两组单位球融合向量为初始化，用真实点击事件的10个曝光候选训练条件选择模型。',
           '七个全局系数先在严格更早的表示上拟合，再固定温度更新单位用户/物品向量；逐ID先验按其曝光次数归一化。保留事件总点击概率。',
           '三档先验、固定0/8/16/32轮快照和三档修正幅度只在 Jan26/25 比较；选定后 Jan24 不再搜索。',
           f'晋升：{passed}；当前最优 [{current["csv"]}](../../{current["csv"]})；0.838 尚未达到。','',
           '| 切分 | 原 AUC | 新 AUC | 差值 | 近分Top1变化 |','|---|---:|---:|---:|---:|']
    for d,a in audits.items():
        e=edges[d];diff=e['after']['near_tie_top1']-e['before']['near_tie_top1']
        lines.append(f'| Jan{d} | {a["old_auc"]:.9f} | {a["new_auc"]:.9f} | {a["delta"]:+.9f} | {100*diff:+.4f} 个百分点 |')
    lines+=['','近分定义：V26 的前两候选 log 概率差 < 0.5，Top1 只统计有点击的事件。',
            '收益很小，多个切分部分查询重叠；固定配方重采样不消除选参偏差。官方评测分数未知。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    readme=ROOT/'README.md';body=readme.read_text();start='<!-- CONDITIONAL_SPHERE_START -->';end='<!-- CONDITIONAL_SPHERE_END -->'
    if start in body:
        a,t=body.split(start,1);_,b=t.split(end,1);body=a+b.lstrip('\n')
    readme.write_text(f'{start}\n本地最优 AUC **{current["validation_26"]["auc"]:.9f}**；目标 **0.838 尚未达到**。\n\n[可提交 CSV]({current["csv"]}) · [候选偏好修正报告](competition/conditional_sphere/REPORT.md) · [历史注意力对照](competition/history_rerank/REPORT.md)\n{end}\n\n'+body)
    write_json(OUT/'state.json',dict(stage='completed',promoted=passed,updated=time.time(),current_csv=current['csv']))
    print(json.dumps(dict(stage='conditional_completed',promoted=passed,validation={d:a['new_auc'] for d,a in audits.items()})),flush=True)


if __name__=='__main__':finish()
