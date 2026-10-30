"""Four causal history rerankers, locked second and third time origins."""
import json
import subprocess
import sys
import time
import numpy as np
from arena_ood import ROOT,guard,write_json,sha
from arena_history_data import OUT
from arena_history_model import CONFIGS
from arena_history_train import metrics
from arena_metric_geometry import preference_probabilities

NAMES=list(CONFIGS)


def command(args,label):
    write_json(OUT/'state.json',dict(stage=label,updated=time.time(),command=args))
    with (OUT/(label+'.log')).open('a') as log:
        subprocess.run([sys.executable,'-u']+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def select():
    from arena_validation import importance_weights,weighted_auc
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text());parent=(ROOT/pointer['csv']).parent
    base={};groups={}
    for day in ['26','25']:
        d=np.load(parent/('validation_'+day+'.npz'));iw,_=importance_weights('2025-01-'+day,d['event_indices'])
        base['auc_'+day]=metrics(d['labels'],d['predictions'],d['predictions'])['auc']
        base['weighted_auc_'+day]=weighted_auc(d['labels'],d['predictions'],iw);groups[day]=(d,iw)
    base['score']=.5*(base['weighted_auc_26']+base['weighted_auc_25']);rows=[]
    for name in NAMES:
        ds={day:np.load(OUT/'runs'/name/('2025-01-'+day)/'validation.npz') for day in groups}
        for day,(d,_) in groups.items():
            for k in ['labels','event_indices']:np.testing.assert_array_equal(ds[day][k],d[k])
        for mass in [.25,.5,1.]:
            r=dict(name=name,mass=mass)
            for day,(d,iw) in groups.items():
                p=preference_probabilities(d['predictions'],ds[day]['correction'],mass)
                m=metrics(d['labels'],p,d['predictions'])
                r['auc_'+day]=m['auc'];r['near_tie_top1_'+day]=m['near_tie_top1']
                r['weighted_auc_'+day]=weighted_auc(d['labels'],p,iw)
            r['score']=.5*(r['weighted_auc_26']+r['weighted_auc_25']);rows.append(r)
    eligible=[r for r in rows if all(r[k]>base[k]+1e-6 for k in ['auc_26','auc_25','score'])]
    selected=None
    if eligible:
        best=max(r['score'] for r in eligible)
        close=[r for r in eligible if r['score']>=best-1e-6]
        selected=max(close,key=lambda r:.5*(r['near_tie_top1_26']+r['near_tie_top1_25']))
    result=dict(baseline=base,selected=selected,eligible=len(eligible),candidates=rows,
                selection_policy='Global AUC and weighted mean must improve both origins. Within1e-6 of best weighted score, prefer better near-tie top1. Frozen before Jan24.')
    write_json(OUT/'selection.json',result)
    return result


def report(selection,stage):
    pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    lines=['# 时间样本外目标历史注意力重排','',
           '第一轮采用固定7轮的244维 long_spectral 精排；第二轮读取候选与历史20个点击在三种向量下的关系、时间和一致性。所有向量与第一轮模型严格早于各自目标。',
           '数值特征只控制注意力权重；残差值来自历史证据。空历史强制零修正，保持事件总点击概率。', '',
           '| 配置 | Jan26 | Jan25 | 固定轮数 |','|---|---:|---:|---:|']
    for n in NAMES:
        a=json.loads((OUT/'runs'/n/'2025-01-26/result.json').read_text());b=json.loads((OUT/'runs'/n/'2025-01-25/result.json').read_text())
        lines.append(f"| {n} | {a['auc']:.9f} | {b['auc']:.9f} | {a['best_epoch']} |")
    lines+=['',f"阶段：{stage}；双切分合格 {selection['eligible']} 组。当前最优 [{pointer['csv']}](../../{pointer['csv']})，本地 AUC {pointer['validation_26']['auc']:.9f}；0.838 未达到。",'',
            '整体AUC与近分首位命中率分别记录，后者在加权AUC相近时用于选择。不将各时间切分视为统计独立。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'protocol.json').exists():write_json(OUT/'protocol.json',dict(configs=CONFIGS,baseline=json.loads((OUT/'incumbent_pointer.json').read_text()),
        epoch_selection='Primary26 only, actual residual transfer onto frozen incumbent, epoch0 included.',
        secondary='Fixed primary epochs on Jan25',third='Only selected recipe on Jan24',target=.838))
    before=guard()
    for name in NAMES:
        command(['arena_history_train.py','--name',name],name+'_26')
        primary=json.loads((OUT/'runs'/name/'2025-01-26/result.json').read_text())
        command(['arena_history_train.py','--name',name,'--cutoff','2025-01-25','--fixed-epochs',str(primary['best_epoch'])],name+'_25')
    selection=select();stage='completed_no_promotion'
    if selection['selected']:
        from arena_push084_audit import compare
        chosen=selection['selected'];name=chosen['name']
        primary=json.loads((OUT/'runs'/name/'2025-01-26/result.json').read_text())
        command(['arena_history_train.py','--name',name,'--cutoff','2025-01-24','--fixed-epochs',str(primary['best_epoch'])],name+'_locked24')
        pointer=json.loads((OUT/'incumbent_pointer.json').read_text());parent=(ROOT/pointer['csv']).parent
        for day in ['26','25','24']:
            d=np.load(parent/('validation_'+day+'.npz'));r=np.load(OUT/'runs'/name/('2025-01-'+day)/'validation.npz')
            p=preference_probabilities(d['predictions'],r['correction'],chosen['mass'])
            new=dict(event_indices=d['event_indices'],labels=d['labels'],predictions=p)
            np.savez_compressed(OUT/('selected_validation_'+day+'.npz'),**new)
            a=compare(d,new,'2025-01-'+day);a['selection_sha256']=sha(OUT/'selection.json');write_json(OUT/('audit_'+day+'.json'),a)
        a=json.loads((OUT/'audit_24.json').read_text());anchor=json.loads((ROOT/'competition/final_v6/audit_24.json').read_text())
        passed=(a['delta']>=-.0001 and a['cohorts']['weekday']['delta']>=-.0001 and a['new_auc']>=anchor['new_auc']-.0001 and
                a['cohorts']['weekday']['new_auc']>=anchor['cohorts']['weekday']['new_auc']-.0001)
        write_json(OUT/'promotion.json',dict(passed=passed,third_delta=a['delta'],target=.838))
        stage='qualified_needs_export' if passed else 'rejected_third_origin'
    else:write_json(OUT/'promotion.json',dict(passed=False,reason='No strictly eligible two-origin correction.'))
    assert guard()==before;report(selection,stage)
    write_json(OUT/'state.json',dict(stage=stage,selected=selection['selected'],updated=time.time()))
    print(json.dumps(dict(stage=stage,selected=selection['selected'])),flush=True)


if __name__=='__main__':run()
