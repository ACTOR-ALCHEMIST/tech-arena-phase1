"""Sequential completion of experiments, held-out audits and accepted delivery."""
import fcntl
import json
import os
import subprocess
import sys
import time
import numpy as np
from arena_ood import ROOT, OUT, RECIPES, guard, write_json
from arena_ood_export import FINAL


def command(args, label):
    write_json(OUT/'delivery_state.json', dict(stage=label, pid=os.getpid(), started=time.time(), command=args))
    print(json.dumps({'stage': label, 'command': args}), flush=True)
    with (OUT/(label+'.log')).open('a') as log:
        subprocess.run([sys.executable, '-u']+args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def report():
    selection = json.loads((FINAL/'selection.json').read_text())
    manifest = json.loads((FINAL/'manifest.json').read_text())
    accepted = json.loads((FINAL/'acceptance.json').read_text())
    chosen, old = selection['selected'], selection['incumbent']
    lines = ['# V4：历史增强与 OOD 鲁棒性实验', '',
             f"完成 {len(selection['candidates'])} 组训练配置及 Jan25 锁定复核。目标为尽量达到主验证 AUC 0.835；本轮主验证实际为 **{chosen['auc_26']:.9f}**。", '',
             '| 配置 | Jan26 AUC | Jan25 AUC | 主验证选中轮数 |', '|---|---:|---:|---:|']
    for row in selection['candidates']:
        lines.append(f"| {row['name']} | {row['primary']['auc']:.9f} | {row['secondary']['auc']:.9f} | {row['primary']['best_epoch']} |")
    lines += ['', '| 最终融合 | Jan26 | Jan25 |', '|---|---:|---:|',
              f"| 原 V3 | {old['auc_26']:.9f} | {old['auc_25']:.9f} |", f"| 本轮 V4 | {chosen['auc_26']:.9f} | {chosen['auc_25']:.9f} |", '',
              '最终权重：'+', '.join(f'{n}={w:g}' for n, w in zip(selection['sources'], chosen['weights']) if w > 0)+'（logit 融合）。', '',
              '增强按事件执行，始终保留 10 个候选和原始标签。删除后的历史与时间戳同步压紧，重算历史长度、时间间隔、重复点击、协同过滤、ALS 折入和 SGNS 历史匹配特征。每轮全部干净训练事件都参与训练，随机一部分输入切换为增强视图。为控制磁盘，增强视图预先存于随机一半事件，训练时重新抽取使用；并非每轮重新采样删除位置。', '',
              '基础表示、聚合统计均来自相应切点之前。Jan26 选择轮数与配置，Jan25 使用同样轮数及学习率计划。Jan24 是权重冻结后的回溯检查，不参与选权重。官方评测集没有标签。', '']
    for cutoff in ['26', '25', '24']:
        path = FINAL/('audit_'+cutoff+'.json')
        if not path.exists():
            continue
        audit = json.loads(path.read_text())
        lines += [f'## Jan{cutoff} 自然群体检查', '', '| 群体 | 事件数 | 原 V3 AUC | 最终 V4 AUC |', '|---|---:|---:|---:|']
        if 'selected_ensemble' in audit['cohorts']:
            for name, a in audit['cohorts']['frozen_v3'].items():
                b = audit['cohorts']['selected_ensemble'][name]
                lines.append(f"| {name} | {a['events']} | {a['auc']:.9f} | {b['auc']:.9f} |")
        lines += ['', f'## Jan{cutoff} 固定 12,000 事件的人工缺失压力测试', '',
                  '| 输入 | 原 V3 AUC | 最终 V4 AUC |', '|---|---:|---:|']
        for name, row in audit['stress'].items():
            if 'selected_ensemble' in row:
                lines.append(f"| {name} | {row['frozen_v3']:.9f} | {row['selected_ensemble']:.9f} |")
        lines += ['', '自然冷用户 AUC 与人工删除历史 AUC 衡量不同情况；压力测试改善不等于真实未来 OOD 必然改善。', '']
    lines += ['## 交付与复现', '', f"提交行数：{accepted['rows']:,}。CSV SHA256：`{accepted['csv_sha256']}`。", '',
              f"独立进程从原始缓存输入与冻结表示重算全部新模型预测，最大误差 {accepted['full_raw_input_new_model_replay_max_error']:.3g}；融合重算最大误差 {accepted['composition_max_error']:.3g}。", '',
              '[最优提交 CSV](../final_v4/submission.csv) · [模型与配方](../final_v4/manifest.json) · [独立验收](../final_v4/acceptance.json)', '',
              '```bash', 'python3 -m unittest -v test_arena_ood.py', 'python3 arena_ood_export.py verify', '```', '',
              '原 V1/V2/V3、Qwen LoRA 结果均保留。训练后的最佳新模型按选定轮数在 Jan24–27 的完整最近窗口 refit，再预测官方评测集。局部增益有模型选择不确定性，不承诺排行榜分数。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    readme = ROOT/'README.md'
    body = readme.read_text()
    start, end = '<!-- OOD_V4_START -->', '<!-- OOD_V4_END -->'
    if start in body:
        body = body.split(end, 1)[1].lstrip('\n')
    block = [start, '# 最新交付：V4 历史增强与 OOD 实验', '',
             f"完成 {len(selection['candidates'])} 组配置及双切点复核。Jan26 AUC **{chosen['auc_26']:.9f}**，Jan25 **{chosen['auc_25']:.9f}**；目标 0.835 {'已达到（本地主验证）' if chosen['auc_26'] >= .835 else '尚未达到'}。", '',
             '[可直接提交的 CSV](competition/final_v4/submission.csv) · [完整实验报告](competition/ood/REPORT.md) · [验收](competition/final_v4/acceptance.json)', '', end, '']
    readme.write_text('\n'.join(block)+'\n'+body)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.delivery.lock').open('a') as delivery:
        fcntl.flock(delivery, fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        if (FINAL/'acceptance.json').exists():
            accepted = json.loads((FINAL/'acceptance.json').read_text())
            pointer = json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
            if accepted.get('passed') and pointer['csv'] == 'competition/final_v4/submission.csv':
                report()
                write_json(OUT/'delivery_state.json', dict(stage='delivery_completed', pid=os.getpid(), finished=time.time(),
                           auc_26=pointer['validation_26']['auc'], target_reached=pointer['validation_26']['auc'] >= .835))
                return
        write_json(OUT/'delivery_state.json', dict(stage='waiting_live_iteration_lock', pid=os.getpid()))
        # Wait for the already-running controller, then run only missing recipes.
        with (OUT/'.iteration.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
        before = guard()
        selection_path = OUT/'selection.json'
        completed_pool = selection_path.exists() and json.loads(selection_path.read_text())['sources'] == ['frozen_v3']+list(RECIPES)
        if not completed_pool:
            command(['arena_ood_iterate.py'], 'complete_pool')
        selection = json.loads((OUT/'selection.json').read_text())
        names = list(RECIPES)
        for cutoff in ['2025-01-26', '2025-01-25']:
            command(['arena_ood_audit.py', '--cutoff', cutoff, '--names']+names, 'audit_'+cutoff[-2:])
        command(['arena_ood.py', 'train', '--name', 'control24', '--cutoff', '2025-01-24', '--fixed-epochs', '4',
                 '--config', 'competition/ood/control24.json'], 'control24')
        active = [name for name, w in zip(selection['sources'][1:], selection['selected']['weights'][1:]) if w > 0]
        for name in active:
            r = json.loads((OUT/'runs'/name/'2025-01-26/result.json').read_text())
            command(['arena_ood.py', 'train', '--name', name, '--cutoff', '2025-01-24', '--fixed-epochs', str(r['best_epoch'])], name+'_locked24')
        command(['arena_ood_audit.py', '--cutoff', '2025-01-24', '--names']+active, 'audit_24')
        command(['arena_ood_export.py', 'export'], 'final_export')
        command(['arena_ood_export.py', 'verify'], 'final_verify')
        assert guard() == before
        command(['arena_ood_export.py', 'promote'], 'final_promote')
        report()
        write_json(OUT/'delivery_state.json', dict(stage='delivery_completed', pid=os.getpid(), finished=time.time(),
                   auc_26=selection['selected']['auc_26'], target_reached=selection['selected']['auc_26'] >= .835))


if __name__ == '__main__':
    write_json(OUT/'control24.json', {'mix': 0})
    run()
