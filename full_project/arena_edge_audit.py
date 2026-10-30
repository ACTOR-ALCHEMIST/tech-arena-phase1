"""Descriptive, fixed-parent audits of borderline recommendation events."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from arena_ood import ROOT,write_json,sha


def measure(labels,old,new,mask):
    y=labels[mask];a=old[mask];b=new[mask];clicked=y.sum(1)>0
    result=dict(events=int(mask.sum()),clicked_events=int(clicked.sum()))
    if y.size and np.unique(y).size==2:
        x=float(roc_auc_score(y.ravel(),a.ravel()));z=float(roc_auc_score(y.ravel(),b.ravel()))
        result.update(old_auc=x,new_auc=z,auc_delta=z-x)
    if clicked.any():
        target=y[clicked].argmax(1);first=a[clicked].argmax(1)==target;second=b[clicked].argmax(1)==target
        corrected=int((~first&second).sum());regressed=int((first&~second).sum())
        result.update(old_top1=float(first.mean()),new_top1=float(second.mean()),top1_delta=float(second.mean()-first.mean()),
                      corrected=corrected,regressed=regressed)
        assert abs(result['top1_delta']-(corrected-regressed)/len(first))<1e-14
    return result


def audit(before,after,out):
    out.mkdir(parents=True,exist_ok=True);rows={}
    for day in ['26','25','24']:
        a=np.load(before/('validation_'+day+'.npz'));b=np.load(after/('validation_'+day+'.npz'))
        for k in ['event_indices','labels']:np.testing.assert_array_equal(a[k],b[k])
        p=a['predictions'];q=b['predictions'];logs=np.sort(np.log(p.clip(1e-8)),axis=1)
        gap=logs[:,-1]-logs[:,-2]
        masks=dict(all_events=np.ones(len(p),bool),near_gap05=gap<.5,near_gap02=gap<.2)
        rows[day]={name:measure(a['labels'],p,q,mask) for name,mask in masks.items()}
        write_json(out/('edge_'+day+'.json'),dict(before=str(before.relative_to(ROOT)),after=str(after.relative_to(ROOT)),
                   cohort_definition='Parent top2 log-probability difference; fixed before reading outcomes. Top1 uses clicked events only.',
                   source_sha256=dict(before=sha(before/('validation_'+day+'.npz')),after=sha(after/('validation_'+day+'.npz'))),cohorts=rows[day]))
    lines=['# 固定原模型边界样本审查','',f'对照：{before.name} → {after.name}。分组依据原模型，结果只作诊断，不用于重新选参数。','',
           '| 日期 | 分组 | 事件数 | 原AUC | 新AUC | Top1变化 | 纠正 / 回退 |','|---|---|---:|---:|---:|---:|---:|']
    for day,groups in rows.items():
        for name,r in groups.items():
            if 'old_auc' not in r or 'old_top1' not in r:continue
            lines.append(f'| Jan{day} | {name} | {r["events"]} | {r["old_auc"]:.9f} | {r["new_auc"]:.9f} | {100*r["top1_delta"]:+.4f} 个百分点 | {r["corrected"]} / {r["regressed"]} |')
    lines+=['','near_gap05 / 02：原模型前两名 log 概率差分别小于0.5 / 0.2。AUC 为该组所有候选行的 ROC AUC，Top1 为有点击事件的首位命中率。',
            '各时间切分部分查询重叠；这些点估计不是统计独立的复现，也不能替代官方评测。','']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    audit(ROOT/a.before,ROOT/a.after,ROOT/a.out)
