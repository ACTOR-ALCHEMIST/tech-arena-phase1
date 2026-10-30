"""Audit a locked latent preference correction, then freeze a directly usable CSV."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from arena_ood import ROOT,guard,sha,write_json
from arena_metric_geometry import cosine,preference_probabilities

OUT=ROOT/'competition/latent_delta_final'
FINAL=ROOT/'competition/final_v26'


def component(cutoff,signal):
    from arena_spherical_repr import model as old_model
    from arena_spherical_long import model as long_model
    from arena_fused_sphere import model as fused_model,fuse
    from arena_core import snapshot
    old=dict(np.load(old_model(cutoff)/'factors_e32.npz'))
    key=signal.removesuffix('_delta')
    if key=='old32':new=old
    elif key in ['long128','long']:
        new=dict(np.load(long_model(cutoff)/'factors_e128.npz'))
    elif key in ['alt128','alt']:
        from arena_spherical_alt import model as alt_model
        new=dict(np.load(alt_model(cutoff)/'factors_e128.npz'))
    elif key in ['fused50','fused75']:
        new=dict(np.load(fused_model(cutoff)/('w'+key[5:]+'.npz')))
    elif key in ['unitfused25','unitfused50','unitfused75']:
        from arena_spherical_alt import model as alt_model
        a=dict(np.load(long_model(cutoff)/'factors_e128.npz'));b=dict(np.load(alt_model(cutoff)/'factors_e128.npz'))
        stats=dict(np.load(snapshot(cutoff)/'stats.npz'))
        new=fuse(a,b,np.linalg.norm(a['users'],axis=1)>0,stats['pop'],int(key[-2:])/100)
    else:raise ValueError(signal)
    return old,new


def values(raw,ix,cutoff,selected):
    old,new=component(cutoff,selected['signal'])
    delta=cosine(new,raw,ix)
    if selected['signal'].endswith('_delta'):delta-=cosine(old,raw,ix)
    return delta


def export(selection,pointer):
    from ctr_data import load_cache
    from arena_finalize import export_csv
    from arena_metric_infer import read_csv
    from arena_core import snapshot
    from run import timestamp
    before=guard();FINAL.mkdir(parents=True,exist_ok=True)
    assert not (FINAL/'manifest.json').exists(),'Use a fresh release directory.'
    shutil.copy2(ROOT/pointer['csv'],FINAL/'frozen_incumbent.csv')
    selected=selection['selected'];old,new=component('2025-01-27',selected['signal'])
    for name,rep in [('old',old),('new',new)]:np.savez(FINAL/(name+'_geometry.npz'),users=rep['users'],items=rep['items'])
    metadata=json.loads((snapshot('2025-01-27')/'metadata.json').read_text())
    assert metadata['max_fit_timestamp']<timestamp('2025-01-27')
    write_json(FINAL/'representation_provenance.json',dict(selected=selected,snapshot=metadata,
               policy='All source representations and their alignment use only data strictly before Jan27; query labels unavailable. Frozen tables need no training cache or optimizer.'))
    _,raw=load_cache(ROOT/'data/cache_v1','evaluation');base=read_csv(FINAL/'frozen_incumbent.csv',raw['rowids'].ravel());parts=[]
    for start in range(0,len(raw['users']),4096):
        ix=np.arange(start,min(start+4096,len(raw['users'])))
        v=cosine(new,raw,ix)
        if selected['signal'].endswith('_delta'):v-=cosine(old,raw,ix)
        parts.append(v)
    delta=np.concatenate(parts);p=preference_probabilities(base,delta,selected['mass'])
    np.save(FINAL/'predictions.npy',p);np.save(FINAL/'correction.npy',delta);export_csv(FINAL/'submission.csv',raw,p)
    for day in ['26','25','24']:
        for source,target in [('selected_validation_'+day+'.npz','validation_'+day+'.npz'),('audit_'+day+'.json','audit_'+day+'.json')]:
            shutil.copy2(OUT/source,FINAL/target)
    shutil.copy2(OUT/'locked_selection.json',FINAL/'selection.json')
    source=FINAL/'source';source.mkdir(exist_ok=True)
    for f in list(ROOT.glob('arena_*.py'))+[ROOT/n for n in ['run.py','ctr_data.py','ctr_model.py','test_arena_latent_delta.py']]:
        shutil.copy2(f,source/f.name)
    manifest=dict(method='conditional_preference_geometry_replacement',selection=selected,subtract_old=selected['signal'].endswith('_delta'),
                  cutoff_timestamp=timestamp('2025-01-27'),target_auc=.838,official_evaluation_labels_available=False,previous_best=pointer,
                  csv=str((FINAL/'submission.csv').relative_to(ROOT)),sha256=sha(FINAL/'submission.csv'),rows=int(p.size),incumbent_digest=before,
                  **{'validation_'+d:dict(auc=json.loads((OUT/('audit_'+d+'.json')).read_text())['new_auc']) for d in ['26','25','24']},
                  artifact_sha256={str(f.relative_to(FINAL)):sha(f) for f in FINAL.rglob('*') if f.is_file() and f.name not in ['manifest.json','acceptance.json']},
                  raw_input_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'data/cache_v1/evaluation').glob('*.npy')})
    write_json(FINAL/'manifest.json',manifest);assert guard()==before
    subprocess.run([sys.executable,'-u',str(source/'arena_latent_delta_infer.py'),'--final',str(FINAL)],cwd=ROOT,check=True)
    assert guard()==before and json.loads((FINAL/'acceptance.json').read_text())['passed']
    write_json(ROOT/'competition/CURRENT_BEST.json',dict(csv=manifest['csv'],sha256=manifest['sha256'],
               manifest=str((FINAL/'manifest.json').relative_to(ROOT)),acceptance=str((FINAL/'acceptance.json').relative_to(ROOT)),
               previous_best_csv=pointer['csv'],**{'validation_'+d:manifest['validation_'+d] for d in ['26','25','24']},
               scope='Best validated local recipe; official evaluation labels unavailable.'))


def report(audits,promoted,selected):
    pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    lines=['# 仅更新候选偏好证据的第二轮修正','',
           '精排保留原来的事件点击概率，第二轮以经过时间验证的新旧向量余弦差修正候选条件概率。单位球向量来自两组独立初始化，融合时用过去物品表做正交对齐。',
           '所有向量的拟合数据严格早于对应查询；修正类型与幅度在 Jan26/25 固定后，再做 Jan24 检查。','',
           f"晋升：{promoted}；当前最优 [{pointer['csv']}](../../{pointer['csv']})；本地 Jan26 AUC {pointer['validation_26']['auc']:.9f}。0.838 尚未达到。",'',
           '| 时间切分 | 原分数 | 新分数 | 差值 |','|---|---:|---:|---:|']
    for day,a in audits.items():lines.append(f"| Jan{day} | {a['old_auc']:.9f} | {a['new_auc']:.9f} | {a['delta']:+.9f} |")
    lines+=['','固定修正：','```json',json.dumps(selected,indent=2),'```','',
            '点估计仍很小；重采样不校正参数选择偏差，多个时间切分部分查询重叠。官方评测标签不可用。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    readme=ROOT/'README.md';body=readme.read_text();start='<!-- LATENT_DELTA_START -->';end='<!-- LATENT_DELTA_END -->'
    if start in body:
        left,tail=body.split(start,1);_,right=tail.split(end,1);body=left+right.lstrip('\n')
    readme.write_text(f"{start}\n# 最新候选偏好修正\n\n晋升：{promoted}；当前最优本地主切分 AUC **{pointer['validation_26']['auc']:.9f}**。目标 0.838 尚未达到。\n\n[直接提交 CSV]({pointer['csv']}) · [本轮报告]({str((OUT/'REPORT.md').relative_to(ROOT))})\n\n{end}\n\n"+body)


def run(probe):
    from ctr_data import load_cache
    from arena_push084_audit import compare
    OUT.mkdir(parents=True,exist_ok=True)
    selected=json.loads((probe/'selection.json').read_text())
    pointer=json.loads((probe/'incumbent_pointer.json').read_text())
    assert selected['selected'] is not None
    assert pointer==json.loads((ROOT/'competition/CURRENT_BEST.json').read_text()),'Re-evaluate the probe against the current incumbent first.'
    before=guard();locked=dict(selected=selected['selected'],source_probe=str(probe.relative_to(ROOT)),source_selection_sha256=sha(probe/'selection.json'),baseline=pointer)
    if (OUT/'locked_selection.json').exists():assert json.loads((OUT/'locked_selection.json').read_text())==locked
    else:write_json(OUT/'locked_selection.json',locked)
    _,raw=load_cache(ROOT/'data/cache_v1','train');parent=(ROOT/pointer['csv']).parent;audits={}
    for day in ['26','25','24']:
        d=np.load(parent/('validation_'+day+'.npz'));delta=values(raw,d['event_indices'],'2025-01-'+day,locked['selected'])
        p=preference_probabilities(d['predictions'],delta,locked['selected']['mass'])
        np.testing.assert_allclose(p.sum(1),d['predictions'].sum(1),atol=2e-7,rtol=1e-6)
        new=dict(event_indices=d['event_indices'],labels=d['labels'],predictions=p)
        np.savez_compressed(OUT/('selected_validation_'+day+'.npz'),**new)
        audit=compare(d,new,'2025-01-'+day);audit['locked_selection_sha256']=sha(OUT/'locked_selection.json');audits[day]=audit
        if day in ['26','25']:
            assert abs(audit['new_auc']-locked['selected']['auc_'+day])<1e-10,'Selected probe and exported geometry disagree.'
        write_json(OUT/('audit_'+day+'.json'),audit)
    third=audits['24'];anchor=json.loads((ROOT/'competition/final_v6/audit_24.json').read_text())
    stable=(third['delta']>=-.0001 and third['cohorts']['weekday']['delta']>=-.0001 and
            third['new_auc']>=anchor['new_auc']-.0001 and third['cohorts']['weekday']['new_auc']>=anchor['cohorts']['weekday']['new_auc']-.0001)
    write_json(OUT/'promotion.json',dict(passed=stable,audit_deltas={d:a['delta'] for d,a in audits.items()},target_auc=.838))
    assert guard()==before
    if stable:export(locked,pointer)
    report(audits,stable,locked['selected'])
    write_json(OUT/'state.json',dict(stage='completed',promoted=stable,updated=time.time(),target_auc=.838))
    print(json.dumps(dict(stage='completed',promoted=stable,audit_deltas={d:a['delta'] for d,a in audits.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',required=True);p.add_argument('--out');p.add_argument('--final');a=p.parse_args()
    if a.out:OUT=ROOT/a.out
    if a.final:FINAL=ROOT/a.final
    run(ROOT/a.probe)
