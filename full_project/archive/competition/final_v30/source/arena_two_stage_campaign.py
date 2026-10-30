"""Run the four chronological reranking ablations, preserve the incumbent."""
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
from arena_ood import ROOT,guard,sha,write_json
from arena_two_stage_data import OUT

NAMES=['residual_mlp','edge_mlp','edge_transformer','edge_transformer_rank']
RUNS=OUT/'runs'
INPUTS=OUT/'inputs'
FINAL=ROOT/'competition/final_v21'


def command(args,label):
    write_json(OUT/'state.json',dict(stage=label,pid=os.getpid(),updated=time.time(),command=args))
    with (OUT/(label+'.log')).open('a') as log:
        subprocess.run([sys.executable,'-u']+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def compose(incumbent,stage1,reranked,mode,mass):
    from arena_finalize import blend
    if mode=='blend':return blend([incumbent,reranked],[1-mass,mass],'logit')
    # Transfer the learned eleven-choice utility correction to the stronger
    # incumbent ensemble. This retains its click/no-click calibration.
    def utilities(p):
        p=np.asarray(p,dtype=np.float64).clip(1e-8,1-1e-8)
        outside=(1-p.sum(1,keepdims=True)).clip(1e-8)
        return np.log(p)-np.log(outside)
    delta=utilities(reranked)-utilities(stage1)
    z=utilities(incumbent)+mass*delta
    logits=np.concatenate([z,np.zeros((len(z),1))],1)
    logits-=logits.max(1,keepdims=True);p=np.exp(logits)
    return (p/p.sum(1,keepdims=True))[:,:10]


def select():
    from sklearn.metrics import roc_auc_score
    from arena_validation import importance_weights,weighted_auc
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text());parent=(ROOT/pointer['csv']).parent
    origins={}
    for day in ['26','25']:
        old=np.load(parent/('validation_'+day+'.npz'))
        baseline=np.load(INPUTS/('valid_2025-01-'+day)/'scores.npz')
        for k in ['labels','event_indices']:np.testing.assert_array_equal(old[k],baseline[k])
        origins[day]=(old,baseline['predictions'],importance_weights('2025-01-'+day,old['event_indices'])[0])
    rows=[]
    for name in NAMES:
        pred={d:np.load(RUNS/name/('2025-01-'+d)/'validation.npz')['predictions'] for d in origins}
        for mode,masses in [('blend',[.05,.1,.25,.5,1.]),('transfer',[.25,.5,.75,1.])]:
            for mass in masses:
                r=dict(name=name,mode=mode,mass=mass)
                for day,(old,stage1,iw) in origins.items():
                    p=compose(old['predictions'],stage1,pred[day],mode,mass)
                    r['auc_'+day]=float(roc_auc_score(old['labels'].ravel(),p.ravel()))
                    r['weighted_auc_'+day]=weighted_auc(old['labels'],p,iw)
                r['score']=.5*(r['weighted_auc_26']+r['weighted_auc_25']);rows.append(r)
    base={}
    for day,(old,_,iw) in origins.items():
        base['auc_'+day]=float(roc_auc_score(old['labels'].ravel(),old['predictions'].ravel()))
        base['weighted_auc_'+day]=weighted_auc(old['labels'],old['predictions'],iw)
    base['score']=.5*(base['weighted_auc_26']+base['weighted_auc_25'])
    eligible=[r for r in rows if all(r[k]>base[k]+1e-6 for k in ['auc_26','auc_25','score'])]
    chosen=max(eligible,key=lambda r:r['score']) if eligible else None
    selection=dict(target_auc=.838,baseline=base,selected=chosen,eligible=len(eligible),candidates=rows,
                   protocol='Primary Jan26 selection, locked epochs on Jan25. Jan24 not used in this search.')
    write_json(OUT/'selection.json',selection)
    if chosen:
        for day,(old,stage1,_) in origins.items():
            p=np.load(RUNS/chosen['name']/('2025-01-'+day)/'validation.npz')['predictions']
            np.savez_compressed(OUT/('selected_validation_'+day+'.npz'),event_indices=old['event_indices'],labels=old['labels'],
                                predictions=compose(old['predictions'],stage1,p,chosen['mode'],chosen['mass']))
    return selection


def report(selection,promoted=False):
    current=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    lines=['# 时间样本外精排＋边缘案例重排','',
           '目标 AUC 0.838。第一轮冻结 200 维特征精排；第二轮输入逐日时间样本外分数、244 维历史特征和候选相似度。',
           '训练第二轮的每个事件，第一轮模型只见过此前日期的标签。第二轮保留 10 个候选和不点击选项；不按标签挑选验证边缘样本。','',
           '| 重排配置 | Jan26 | Jan25 | 主切分轮数 |', '|---|---:|---:|---:|']
    for name in NAMES:
        a=json.loads((RUNS/name/'2025-01-26/result.json').read_text())
        b=json.loads((RUNS/name/'2025-01-25/result.json').read_text())
        lines.append(f"| {name} | {a['auc']:.9f} | {b['auc']:.9f} | {a['best_epoch']} |")
    lines+=['',f"双切分合格融合数：{selection['eligible']}；晋升：{promoted}。",f"当前最优：[{current['csv']}](../../{current['csv']})；本地 AUC {current['validation_26']['auc']:.9f}。",'',
            '候选之间首位命中率、接近打平样本 AUC 见各 result.json 的 cohorts 字段。全局 AUC 决定是否替换提交。',
            'Jan24/25/26 的后续查询部分重叠，不能将它们当成三个统计独立的验证集。尚无官方评测标签。','']
    if selection['selected']:lines+=['```json',json.dumps(selection['selected'],indent=2),'```','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    readme=ROOT/'README.md';body=readme.read_text();start='<!-- TWO_STAGE_START -->';end='<!-- TWO_STAGE_END -->'
    if start in body:
        left,tail=body.split(start,1);_,right=tail.split(end,1);body=left+right.lstrip('\n')
    block=f"{start}\n# 冲分目标 0.838：精排＋重排\n\n{len(NAMES)} 组时间样本外重排实验已完成。双切分合格融合 {selection['eligible']} 组；晋升：{promoted}。当前最优本地 AUC **{current['validation_26']['auc']:.9f}**。\n\n[当前提交 CSV]({current['csv']}) · [实验报告]({str((OUT/'REPORT.md').relative_to(ROOT))})\n\n{end}\n\n"
    readme.write_text(block+body)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'incumbent.json').exists():
        write_json(OUT/'incumbent.json',guard())
        write_json(OUT/'incumbent_pointer.json',json.loads((ROOT/'competition/CURRENT_BEST.json').read_text()))
    assert guard()==json.loads((OUT/'incumbent.json').read_text())
    for name in NAMES:
        command(['arena_two_stage.py','--name',name],name+'_26')
        primary=json.loads((RUNS/name/'2025-01-26/result.json').read_text())
        command(['arena_two_stage.py','--name',name,'--cutoff','2025-01-25','--fixed-epochs',str(primary['best_epoch'])],name+'_25')
    selection=select();promoted=False
    if selection['selected']:
        chosen=selection['selected'];name=chosen['name']
        primary=json.loads((RUNS/name/'2025-01-26/result.json').read_text())
        command(['arena_two_stage.py','--name',name,'--cutoff','2025-01-24','--fixed-epochs',str(primary['best_epoch'])],name+'_locked24')
        from arena_push084_audit import compare
        pointer=json.loads((OUT/'incumbent_pointer.json').read_text());parent=(ROOT/pointer['csv']).parent
        for day in ['26','25']:
            audit=compare(np.load(parent/('validation_'+day+'.npz')),np.load(OUT/('selected_validation_'+day+'.npz')),'2025-01-'+day)
            audit['selection_sha256']=sha(OUT/'selection.json');write_json(OUT/('audit_'+day+'.json'),audit)
        old=np.load(parent/'validation_24.npz');rank=np.load(INPUTS/'valid_2025-01-24/scores.npz')
        new=np.load(RUNS/name/'2025-01-24/validation.npz')
        for k in ['labels','event_indices']:
            np.testing.assert_array_equal(old[k],rank[k]);np.testing.assert_array_equal(old[k],new[k])
        p=compose(old['predictions'],rank['predictions'],new['predictions'],chosen['mode'],chosen['mass'])
        result=dict(event_indices=old['event_indices'],labels=old['labels'],predictions=p)
        np.savez_compressed(OUT/'selected_validation_24.npz',**result)
        audit=compare(old,result,'2025-01-24');audit['selection_sha256']=sha(OUT/'selection.json');write_json(OUT/'audit_24.json',audit)
        anchor=json.loads((ROOT/'competition/final_v6/audit_24.json').read_text())
        stable=(audit['delta']>=-.0001 and audit['cohorts']['weekday']['delta']>=-.0001 and
                audit['new_auc']>=anchor['new_auc']-.0001 and audit['cohorts']['weekday']['new_auc']>=anchor['cohorts']['weekday']['new_auc']-.0001)
        write_json(OUT/'promotion.json',dict(passed=stable,third_origin=audit,absolute_anchor='V6',tolerance=.0001))
        if stable:
            command(['arena_two_stage_export.py','--campaign',str(OUT),'--final',str(FINAL)],'final_export_and_verify')
            promoted=True
    else:write_json(OUT/'promotion.json',dict(passed=False,reason='No configuration improved both chronological AUCs and weighted selection score.'))
    report(selection,promoted)
    write_json(OUT/'state.json',dict(stage='completed',promoted=promoted,updated=time.time(),target_auc=.838))
    print(json.dumps(dict(stage='completed',selected=selection['selected'],promoted=promoted)),flush=True)


if __name__=='__main__':run()
