"""Lock the metric correction on a third origin, freeze and replay a CSV."""
import json
import shutil
import subprocess
import sys
import time
import numpy as np
from arena_ood import ROOT,guard,sha,write_json
import arena_metric_rerank as metric

OUT=ROOT/'competition/metric_oof'
FINAL=ROOT/'competition/final_v23'
metric.OUT=OUT;metric.GEOMETRY_MODE='aligned_oof'


def report(promoted):
    pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    selected=json.loads((OUT/'selection.json').read_text())['selected']
    lines=['# 按时间生成向量的低参数匹配修正','',
           '训练第一轮分数和向量都严格早于各自训练目标。向量先对齐到 Jan21 参考，再以 192 个参数学习用户、最近三次点击和衰减历史到候选的匹配变换。只改变条件候选概率，保持每事件总点击概率。',
           '六组固定配置：1/3 个用户侧向量、三档岭正则；第二与第三时间切分不重新搜索优化器迭代次数。', '',
           f'晋升：{promoted}；当前提交 [{pointer["csv"]}](../../{pointer["csv"]})；本地 Jan26 AUC {pointer["validation_26"]["auc"]:.9f}。目标 0.838 尚未达到。','']
    if selected:lines+=['```json',json.dumps(selected,indent=2),'```','']
    for day in ['26','25','24']:
        p=OUT/('audit_'+day+'.json')
        if p.exists():
            a=json.loads(p.read_text());lines.append(f"Jan{day}: {a['old_auc']:.9f} → {a['new_auc']:.9f} ({a['delta']:+.9f})。")
    lines+=['','提升是本地小幅点估计；重采样不校正配置搜索偏差，各时间切分部分查询重叠。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    readme=ROOT/'README.md';body=readme.read_text();start='<!-- METRIC_OOF_START -->';end='<!-- METRIC_OOF_END -->'
    if start in body:
        l,t=body.split(start,1);_,r=t.split(end,1);body=l+r.lstrip('\n')
    block=f"{start}\n# 最新时间样本外匹配修正\n\n新增 6 组小参数匹配实验；晋升：{promoted}。当前最优本地主切分 AUC **{pointer['validation_26']['auc']:.9f}**；目标 **0.838 尚未达到**。\n\n[最优提交 CSV]({pointer['csv']}) · [本轮报告](competition/metric_oof/REPORT.md)\n\n{end}\n\n"
    readme.write_text(block+body)


def export(selection):
    from ctr_data import load_cache
    from arena_aligned_repr import model as representation
    from arena_finalize import export_csv
    from arena_metric_geometry import profiles,correction,preference_probabilities
    from arena_metric_infer import read_csv
    before=guard();assert before==json.loads((OUT/'incumbent_digest.json').read_text())
    assert not (FINAL/'manifest.json').exists(),'Use a fresh version after a completed export.'
    FINAL.mkdir(parents=True,exist_ok=True)
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text())
    shutil.copy2(ROOT/pointer['csv'],FINAL/'frozen_incumbent.csv')
    selected=selection['selected'];index=selected['index'];fit=metric.fit('2025-01-27',index,load_cache(ROOT/'data/cache_v1','train')[1])
    shutil.copy2(fit/'metric.npz',FINAL/'metric.npz');shutil.copy2(fit/'result.json',FINAL/'fit.json')
    rep=representation('2025-01-27');np.savez(FINAL/'representation.npz',**rep)
    _,raw=load_cache(ROOT/'data/cache_v1','evaluation');base=read_csv(FINAL/'frozen_incumbent.csv',raw['rowids'].ravel())
    matrix=np.load(FINAL/'metric.npz')['matrix'];parts=[]
    for start in range(0,len(raw['users']),4096):
        ix=np.arange(start,min(start+4096,len(raw['users'])))
        a,b=profiles(raw,ix,rep);parts.append(correction(a,b,matrix))
    delta=np.concatenate(parts);p=preference_probabilities(base,delta,selected['mass'])
    np.save(FINAL/'correction.npy',delta);np.save(FINAL/'predictions.npy',p);export_csv(FINAL/'submission.csv',raw,p)
    for day in ['26','25','24']:
        shutil.copy2(OUT/('selected_validation_'+day+'.npz'),FINAL/('validation_'+day+'.npz'))
        shutil.copy2(OUT/('audit_'+day+'.json'),FINAL/('audit_'+day+'.json'))
    shutil.copy2(OUT/'selection.json',FINAL/'selection.json')
    source=FINAL/'source';source.mkdir(exist_ok=True)
    # Include training provenance and every transitive root module; replay itself
    # needs only the small NumPy geometry, ctr_data and the original CSV checker.
    for file in list(ROOT.glob('arena_*.py'))+[ROOT/n for n in ['run.py','ctr_data.py','ctr_model.py','test_arena_metric_rerank.py']]:
        shutil.copy2(file,source/file.name)
    if (OUT/'tests.log').exists():shutil.copy2(OUT/'tests.log',FINAL/'tests.log')
    manifest=dict(method='past_only_aligned_OOF_conditional_metric_correction',selection=selected,target_auc=.838,
                  previous_best=pointer,official_evaluation_labels_available=False,rows=int(p.size),
                  csv=str((FINAL/'submission.csv').relative_to(ROOT)),sha256=sha(FINAL/'submission.csv'),incumbent_digest=before,
                  artifact_sha256={str(f.relative_to(FINAL)):sha(f) for f in FINAL.rglob('*') if f.is_file() and f.name not in ['manifest.json','acceptance.json']},
                  raw_input_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'data/cache_v1/evaluation').glob('*.npy')},
                  **{'validation_'+d:dict(auc=json.loads((OUT/('audit_'+d+'.json')).read_text())['new_auc']) for d in ['26','25','24']})
    write_json(FINAL/'manifest.json',manifest);assert guard()==before
    subprocess.run([sys.executable,'-u',str(source/'arena_metric_infer.py'),'--final',str(FINAL)],cwd=ROOT,check=True)
    assert guard()==before and json.loads((FINAL/'acceptance.json').read_text())['passed']
    write_json(ROOT/'competition/CURRENT_BEST.json',dict(csv=manifest['csv'],sha256=manifest['sha256'],
               manifest=str((FINAL/'manifest.json').relative_to(ROOT)),acceptance=str((FINAL/'acceptance.json').relative_to(ROOT)),
               previous_best_csv=pointer['csv'],**{'validation_'+d:manifest['validation_'+d] for d in ['26','25','24']},
               scope='Best validated local recipe; official evaluation labels unavailable.'))


def run():
    from ctr_data import load_cache
    from arena_push084_audit import compare
    if not (OUT/'selection.json').exists():metric.run()
    selection=json.loads((OUT/'selection.json').read_text());selected=selection['selected']
    if selected is None:report(False);return
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text());parent=(ROOT/pointer['csv']).parent
    _,raw=load_cache(ROOT/'data/cache_v1','train');reports={}
    for day in ['26','25','24']:
        d=np.load(parent/('validation_'+day+'.npz'))
        delta=metric.correction(raw,d['event_indices'],'2025-01-'+day,selected['index'])
        p=metric.preference_probabilities(d['predictions'],delta,selected['mass'])
        new=dict(event_indices=d['event_indices'],labels=d['labels'],predictions=p)
        np.savez_compressed(OUT/('selected_validation_'+day+'.npz'),**new)
        audit=compare(d,new,'2025-01-'+day);audit['selection_sha256']=sha(OUT/'selection.json')
        write_json(OUT/('audit_'+day+'.json'),audit);reports[day]=audit
    third=reports['24'];anchor=json.loads((ROOT/'competition/final_v6/audit_24.json').read_text())
    stable=(third['delta']>=-.0001 and third['cohorts']['weekday']['delta']>=-.0001 and
            third['new_auc']>=anchor['new_auc']-.0001 and third['cohorts']['weekday']['new_auc']>=anchor['cohorts']['weekday']['new_auc']-.0001)
    write_json(OUT/'promotion.json',dict(passed=stable,third_delta=third['delta'],anchor='V6',tolerance=.0001))
    if stable:export(selection)
    report(stable);write_json(OUT/'state.json',dict(stage='completed',promoted=stable,updated=time.time(),target_auc=.838))
    print(json.dumps(dict(stage='completed',promoted=stable,audit_deltas={d:a['delta'] for d,a in reports.items()})),flush=True)


if __name__=='__main__':run()
