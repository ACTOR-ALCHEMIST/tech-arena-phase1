"""Audit, freeze, replay and promote only the locked event-mass recipe."""
import json
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from arena_ood import ROOT,write_json,sha,guard
from arena_event_mass import OUT,RECIPE,train,apply,inputs

FINAL=ROOT/'competition/final_v28'


def export(selected,pointer):
    from arena_history_data import fit_first
    from arena_core import snapshot
    from arena_push084_export import representation_ignore
    from arena_event_mass_infer import replay
    from arena_finalize import export_csv
    from ctr_data import load_cache
    from run import timestamp
    before=guard();assert not (FINAL/'manifest.json').exists(),'Use a new release folder.'
    torch.set_num_threads(2);_,raw=load_cache(ROOT/'data/cache_v1','train')
    calibrator=train('2025-01-27',selected['ridge'],raw);first=fit_first('2025-01-27')
    FINAL.mkdir(parents=True,exist_ok=True);(FINAL/'first_model').mkdir(exist_ok=True)
    np.savez(FINAL/'calibrator.npz',**calibrator);shutil.copy2(first/'model.pt',FINAL/'first_model/model.pt')
    first_recipe=json.loads((first/'result.json').read_text());write_json(FINAL/'first_recipe.json',first_recipe)
    assert first_recipe['max_fit_timestamp']<timestamp('2025-01-27')
    final_protocol=OUT/'2025-01-27'/('ridge'+str(selected['ridge']))/'protocol.json'
    protocol=json.loads(final_protocol.read_text());assert protocol['max_fit_timestamp']<timestamp('2025-01-27')
    shutil.copy2(final_protocol,FINAL/'calibration_protocol.json')
    shutil.copy2(ROOT/pointer['csv'],FINAL/'frozen_incumbent.csv')
    shutil.copytree(snapshot('2025-01-27'),FINAL/'representations/snapshot_2025-01-27',dirs_exist_ok=True,ignore=representation_ignore)
    metadata=json.loads((snapshot('2025-01-27')/'metadata.json').read_text());assert metadata['max_fit_timestamp']<timestamp('2025-01-27')
    # Inference needs only this provisional selection before artifact hashes exist.
    write_json(FINAL/'manifest.json',dict(selection=selected))
    _,evaluation=load_cache(ROOT/'data/cache_v1','evaluation');p,delta=replay(FINAL,evaluation)
    np.save(FINAL/'predictions.npy',p);np.save(FINAL/'correction.npy',delta);export_csv(FINAL/'submission.csv',evaluation,p)
    for d in ['26','25','24']:
        shutil.copy2(OUT/('selected_validation_'+d+'.npz'),FINAL/('validation_'+d+'.npz'))
        shutil.copy2(OUT/('audit_'+d+'.json'),FINAL/('audit_'+d+'.json'))
    shutil.copy2(OUT/'selection.json',FINAL/'selection.json');write_json(FINAL/'recipe.json',RECIPE)
    source=FINAL/'source';source.mkdir(exist_ok=True)
    for f in list(ROOT.glob('arena_*.py'))+[ROOT/n for n in ['run.py','ctr_data.py','ctr_model.py','test_arena_event_mass.py']]:shutil.copy2(f,source/f.name)
    manifest=dict(method='factorized_event_click_mass_time_oof_calibration',selection=selected,recipe=RECIPE,
                  formula='p_new = p_parent / sum(p_parent) * sigmoid(logit(sum(p_parent)) + mass * residual)',
                  cutoff_timestamp=timestamp('2025-01-27'),target_auc=.838,official_evaluation_labels_available=False,
                  previous_best=pointer,csv=str((FINAL/'submission.csv').relative_to(ROOT)),sha256=sha(FINAL/'submission.csv'),
                  rows=int(p.size),incumbent_digest=before,
                  **{'validation_'+d:dict(auc=json.loads((OUT/('audit_'+d+'.json')).read_text())['new_auc']) for d in ['26','25','24']},
                  artifact_sha256={str(f.relative_to(FINAL)):sha(f) for f in FINAL.rglob('*') if f.is_file() and f.name not in ['manifest.json','acceptance.json']},
                  raw_input_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'data/cache_v1/evaluation').glob('*.npy')})
    write_json(FINAL/'manifest.json',manifest);assert guard()==before
    cmd=[sys.executable,'-u',str(source/'arena_event_mass_infer.py'),'--final',str(FINAL)]
    subprocess.run(cmd,check=True,cwd=ROOT)
    assert guard()==before and json.loads((FINAL/'acceptance.json').read_text())['passed']
    write_json(ROOT/'competition/CURRENT_BEST.json',dict(csv=manifest['csv'],sha256=manifest['sha256'],
               manifest=str((FINAL/'manifest.json').relative_to(ROOT)),acceptance=str((FINAL/'acceptance.json').relative_to(ROOT)),
               previous_best_csv=pointer['csv'],**{'validation_'+d:manifest['validation_'+d] for d in ['26','25','24']},
               scope='Best validated local recipe; official evaluation labels unavailable.'))
    subprocess.run(cmd+['--check-only'],check=True,cwd=ROOT)


def finish():
    from ctr_data import load_cache
    from arena_push084_audit import compare
    selected=json.loads((OUT/'selection.json').read_text())['selected']
    assert selected is not None
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text());assert pointer==json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    before=guard();torch.set_num_threads(2);_,raw=load_cache(ROOT/'data/cache_v1','train')
    parent=(ROOT/pointer['csv']).parent;audits={}
    for d in ['26','25','24']:
        if d=='24' and (OUT/'audit_24.json').exists() and (OUT/'selected_validation_24.npz').exists():
            audits[d]=json.loads((OUT/'audit_24.json').read_text());continue
        baseline=np.load(parent/('validation_'+d+'.npz'))
        score=np.load(OUT/('2025-01-'+d)/('ridge'+str(selected['ridge']))/'validation.npz')
        for k in ['event_indices','labels']:np.testing.assert_array_equal(score[k],baseline[k])
        p=apply(baseline['predictions'],score['correction'],selected['mass'])
        new=dict(event_indices=score['event_indices'],labels=score['labels'],predictions=p)
        np.savez_compressed(OUT/('selected_validation_'+d+'.npz'),**new)
        audits[d]=compare(baseline,new,'2025-01-'+d)
    for d,a in audits.items():
        a['locked_selection_sha256']=sha(OUT/'selection.json');write_json(OUT/('audit_'+d+'.json'),a)
        if d in ['26','25']:assert abs(a['new_auc']-selected['auc_'+d])<1e-12
    third=audits['24'];anchor=json.loads((ROOT/'competition/final_v6/audit_24.json').read_text())
    passed=(third['delta']>=-.0001 and third['cohorts']['weekday']['delta']>=-.0001 and
            third['new_auc']>=anchor['new_auc']-.0001 and third['cohorts']['weekday']['new_auc']>=anchor['cohorts']['weekday']['new_auc']-.0001)
    write_json(OUT/'promotion.json',dict(passed=passed,audit_deltas={d:a['delta'] for d,a in audits.items()},target_auc=.838))
    assert guard()==before
    if passed:export(selected,pointer)
    current=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    with (OUT/'REPORT.md').open('a') as f:
        f.write('\n固定第三切分：\n\n')
        for d,a in audits.items():f.write(f'Jan{d}: {a["old_auc"]:.9f} → {a["new_auc"]:.9f} ({a["delta"]:+.9f})。\n\n')
        f.write(f'晋升：{passed}；当前最优 [{current["csv"]}](../../{current["csv"]})。三个切分非统计独立，提升为小幅点估计。\n')
    readme=ROOT/'README.md';body=readme.read_text();start='<!-- EVENT_MASS_START -->';end='<!-- EVENT_MASS_END -->'
    if start in body:
        a,t=body.split(start,1);_,b=t.split(end,1);body=a+b.lstrip('\n')
    readme.write_text(f'{start}\n当前可提交最优：[{current["csv"]}]({current["csv"]})。本地 Jan26 AUC **{current["validation_26"]["auc"]:.9f}**；目标 **0.838 尚未达到**。\n\n[事件点击概率修正报告](competition/event_mass/REPORT.md) · [真实曝光条件重排](competition/conditional_sphere/REPORT.md) · [近分前两名对照](competition/borderline_pair/REPORT.md)\n{end}\n\n'+body)
    write_json(OUT/'state.json',dict(stage='completed',promoted=passed,updated=time.time(),current_csv=current['csv']))
    print(json.dumps(dict(stage='mass_finalized',promoted=passed,auc={d:a['new_auc'] for d,a in audits.items()})),flush=True)


if __name__=='__main__':finish()
