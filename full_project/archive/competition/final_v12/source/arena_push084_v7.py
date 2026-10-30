"""A fresh campaign pinned to accepted V6: residual capacity and more origins."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from arena_ood import ROOT, guard, sha, write_json

OUT = ROOT/'competition/push084_v7'
FINAL = ROOT/'competition/final_v7'
NAMES = ['v7_residual8_wide', 'v7_residual24_wide', 'v7_frozen_graph16_wide',
         'v7_daily_expanding_wide', 'v7_origin_expanding_wide', 'v7_origin_expanding_cross768']
VERIFY_SCRIPT = 'arena_push084_v7.py'
EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'test_arena_v7.py']
DESCRIPTION = '冻结 SGNS8 后增加图初始化低秩偏好残差、固定图物品表示后拟合用户偏好，以及从 Jan21 起扩展滚动训练数据。'


def campaign_module():
    import arena_push084 as campaign
    campaign.OUT, campaign.NAMES = OUT, NAMES
    return campaign


def audit():
    from arena_push084_audit import compare
    from arena_finalize import blend
    campaign = campaign_module()
    s = json.loads((OUT/'selection.json').read_text())
    pointer = json.loads((OUT/'incumbent_pointer.json').read_text())
    parent = (ROOT/pointer['csv']).parent
    reports = {}
    for day in ['26', '25']:
        old = np.load(parent/('validation_'+day+'.npz'))
        new = np.load(OUT/('selected_validation_'+day+'.npz'))
        reports[day] = compare(old, new, '2025-01-'+day)
        reports[day]['selection_sha256'] = sha(OUT/'selection.json')
        write_json(OUT/('audit_'+day+'.json'), reports[day])
    if not s['eligible']:
        return reports
    old = np.load(parent/'validation_24.npz')
    parts, weights = [old['predictions']], [s['selected']['weights'][0]]
    for name, weight in zip(s['sources'][1:], s['selected']['weights'][1:]):
        if weight <= 0:
            continue
        if name == 'refreshed_mf':
            # Existing native head and the fixed refreshed recipe were evaluated
            # in the previous campaign; construct the same prior-only input.
            from arena_robust_mf import load_model, refresh_model, predict
            from ctr_data import load_cache
            import torch
            torch.set_num_threads(2)
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            path = ROOT/'competition/robust_mf/runs/causal_frozen/2025-01-24'
            _, raw = load_cache(ROOT/'data/cache_v1', 'train')
            p = predict(refresh_model(load_model(path, device), '2025-01-24', device), raw, old['event_indices'], device)
        else:
            primary, _ = campaign.record(name, '2025-01-26')
            campaign.command(['arena_ood.py', 'train', '--name', name, '--config', 'configs/'+name+'.json',
                              '--cutoff', '2025-01-24', '--fixed-epochs', str(primary['best_epoch'])], name+'_locked24')
            _, d = campaign.record(name, '2025-01-24')
            for key in ['event_indices', 'labels']:
                np.testing.assert_array_equal(d[key], old[key])
            p = d['predictions']
        parts.append(p)
        weights.append(weight)
    new = dict(event_indices=old['event_indices'], labels=old['labels'], predictions=blend(parts, weights, 'logit'))
    np.savez_compressed(OUT/'selected_validation_24.npz', **new)
    reports['24'] = compare(old, new, '2025-01-24')
    reports['24'].update(selection_sha256=sha(OUT/'selection.json'), protocol='Weights, parameterization and epoch counts frozen before this third origin.')
    write_json(OUT/'audit_24.json', reports['24'])
    return reports


def report(reports, promoted):
    s = json.loads((OUT/'selection.json').read_text())
    pointer = json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    incumbent = json.loads((OUT/'incumbent_pointer.json').read_text())
    lines = ['# 继续向 0.84：'+OUT.name, '',
             f"不可变基线：{incumbent['csv']}。{DESCRIPTION}所有拟合目标在各自切分之前。", '',
             '| 配置 | Jan26 | Jan25 | 轮数 |', '|---|---:|---:|---:|']
    for r in s['candidates']:
        lines.append(f"| {r['name']} | {r['primary']['auc']:.9f} | {r['secondary']['auc']:.9f} | {r['primary']['best_epoch']} |")
    lines += ['', f"晋升新版本：{promoted}。当前最优 CSV：[{pointer['csv']}](../../{pointer['csv']})。", '',
              f"本地主切分 {pointer['validation_26']['auc']:.9f}；0.84 {'已达到（本地）' if pointer['validation_26']['auc'] >= .84 else '尚未达到'}。官方标签不可用。", '']
    for day, row in reports.items():
        lines.append(f"Jan{day}：{row['old_auc']:.9f} → {row['new_auc']:.9f}（{row['delta']:+.9f}）。")
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    marker = OUT.name.upper()
    readme = ROOT/'README.md'
    body = readme.read_text()
    if '<!-- '+marker+'_START -->' in body:
        left, tail = body.split('<!-- '+marker+'_START -->', 1)
        _, right = tail.split('<!-- '+marker+'_END -->', 1)
        body = left+right.lstrip('\n')
    block = [f'<!-- {marker}_START -->', '# 继续向 0.84 优化', '',
             f"新增 {len(NAMES)} 组配置；晋升：{promoted}。当前最优本地 Jan26 AUC **{pointer['validation_26']['auc']:.9f}**，Jan25 **{pointer['validation_25']['auc']:.9f}**；0.84 {'已达到（本地）' if pointer['validation_26']['auc'] >= .84 else '尚未达到'}。", '',
             f"[当前最优 CSV]({pointer['csv']}) · [本轮报告]({str((OUT/'REPORT.md').relative_to(ROOT))})", '',
             f'<!-- {marker}_END -->', '']
    readme.write_text('\n'.join(block)+'\n'+body)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    campaign = campaign_module()
    campaign.train()
    import arena_push084_reblend as refine
    refine.OUT = OUT
    refine.run()
    before = guard()
    reports = audit()
    s = json.loads((OUT/'selection.json').read_text())
    stable = s['eligible'] > 0 and reports['24']['delta'] >= -.0001 and reports['24']['cohorts']['weekday']['delta'] >= -.0001
    # Repeated small promotions must not accumulate an ever-larger OOD loss.
    anchor_path = ROOT/'competition/final_v6/audit_24.json'
    anchor_check = None
    if anchor_path.exists() and '24' in reports:
        anchor = json.loads(anchor_path.read_text())
        anchor_check = dict(origin='V6', tolerance=.0001,
                            overall_floor=anchor['new_auc']-.0001,
                            weekday_floor=anchor['cohorts']['weekday']['new_auc']-.0001)
        anchor_check['passed'] = (reports['24']['new_auc'] >= anchor_check['overall_floor'] and
                                  reports['24']['cohorts']['weekday']['new_auc'] >= anchor_check['weekday_floor'])
        stable = stable and anchor_check['passed']
    write_json(OUT/'promotion.json', dict(promoted=bool(stable), local_dual_origin_eligible=s['eligible'] > 0,
               third_origin_tolerance=.0001, absolute_ood_anchor=anchor_check))
    assert guard() == before
    if stable:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = OUT, FINAL
        exporter.export(verify_script=VERIFY_SCRIPT, extra_sources=EXTRA_SOURCES)
    report(reports, bool(stable))
    write_json(OUT/'state.json', dict(stage='completed', promoted=bool(stable), updated=time.time()))
    print(json.dumps(dict(stage='completed', promoted=bool(stable), selected=s['selected'],
                          audits={d: r['delta'] for d, r in reports.items()})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    if a.verify:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = OUT, FINAL
        exporter.verify()
    else:
        run()
