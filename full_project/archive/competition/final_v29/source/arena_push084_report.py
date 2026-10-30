"""Record measured outcomes and the accepted CSV without overstating the target."""
import json
from arena_ood import ROOT, write_json
from arena_push084 import OUT


def run():
    selection = json.loads((OUT/'selection.json').read_text())
    pointer = json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    old, new = selection['incumbent'], selection['selected']
    lines = ['# 向 0.84 迭代的实验记录', '',
             '本轮在 Apple Silicon 本机运行。所有表示只使用对应快照之前的点击或曝光；评分头训练于之后、验证切分之前的事件。所有候选保留完整十个物品及原始标签。未使用官方评测标签，未上传提交。', '',
             '测试方向包括：历史相邻点击的共享物品表示、真实曝光与历史点击联合训练、逐日首次请求训练、用户均衡和跨事件排序损失、稀疏图的 8／16／32／64 维表示、条件选择与点击率分支分离、用户截距及工作日／周末偏好。', '',
             '图表示还暴露了归一化问题：原实现按绝对方差把四个有变化但量级很小的匹配值视为常量。新配置用相对数值范围识别常量，检查点保存各自的均值与尺度；原有模型不改动。', '',
             '| 配置 | Jan26 AUC | Jan25 AUC | 主切分选定轮数 | 评分头参数 |',
             '|---|---:|---:|---:|---:|']
    for row in selection['candidates']:
        a, b = row['primary'], row['secondary']
        lines.append(f"| {row['name']} | {a['auc']:.9f} | {b['auc']:.9f} | {a['best_epoch']} | {a['parameters']:,} |")
    lines += ['', '## 选择方法', '',
              '真实评测的首次请求可延后 7 天，而主验证 Jan26 只覆盖 24 小时。新增 origin 配方从多个更早快照预测该用户的首次后续请求，保留真实时间、历史与标签；相同拟合目标可有多个信息快照视图。另单独报告请求延迟和工作日／周末群体。', '',
              '主切分 Jan26 选择训练轮数；Jan25 固定相同轮数和学习率日程。候选融合在两个切分的原始 AUC 和平均群体加权 AUC 上均需提高。权重在 Jan24 第三切分之前冻结，第三切分用于检查更早时期及工作日表现。', '',
              '滚动切分会共享部分目标事件，不能把两个切分当成统计独立实验。Jan24 的工作日群体只包含 Jan24 当天，和 Jan25／Jan26 的目标日期不重合。', '',
              '| 配方 | Jan26 | Jan25 |', '|---|---:|---:|',
              f"| 本轮前最优 | {old['auc_26']:.9f} | {old['auc_25']:.9f} |",
              f"| 本轮选中 | {new['auc_26']:.9f} | {new['auc_25']:.9f} |", '',
              '选中 logit 融合权重：', '']
    for name, weight in zip(selection['sources'], new['weights']):
        if weight > 0:
            lines.append(f'- `{name}`：{weight:.8f}')
    lines += ['', '## 时间复核与 OOD', '']
    for day in ['26', '25', '24']:
        path = OUT/('audit_'+day+'.json')
        if path.exists():
            d = json.loads(path.read_text())
            lines += [f"### Jan{day}", '', f"整体 AUC：{d['old_auc']:.9f} → {d['new_auc']:.9f}，差值 {d['delta']:+.9f}。", '',
                      '| 群体 | 事件数 | 原 AUC | 新 AUC | 差值 |', '|---|---:|---:|---:|---:|']
            for name, c in d['cohorts'].items():
                lines.append(f"| {name} | {c['events']:,} | {c['old_auc']:.6f} | {c['new_auc']:.6f} | {c['delta']:+.6f} |")
            ci = d['event_poisson_bootstrap']['delta_percentile95']
            lines += ['', f"固定所选配方，按完整事件进行 200 次 Poisson 重采样；AUC 差值的 95% 分位区间 [{ci[0]:+.7f}, {ci[1]:+.7f}]。这不校正多轮试验的选择偏差，小群体结果波动更大。", '']
    stress = OUT/'stress.json'
    if stress.exists():
        d = json.loads(stress.read_text())
        lines += ['## 固定历史缺失压力测试', '', '固定 12,000 个完整事件，保持原始标签。此测试不用于挑选权重，也不等于未来真实 OOD。', '',
                  '| 输入 | 原 AUC | 新 AUC | 差值 |', '|---|---:|---:|---:|']
        for name, r in d['stress'].items():
            lines.append(f"| {name} | {r['old_auc']:.6f} | {r['new_auc']:.6f} | {r['delta']:+.6f} |")
        lines += ['']
    lines += ['## 提交文件', '', f"当前接受的最优 CSV：[{pointer['csv']}](../../{pointer['csv']})。", '',
              f"本地主切分 AUC **{pointer['validation_26']['auc']:.9f}**；目标 **0.84 {'已达到（仅本地主切分）' if pointer['validation_26']['auc'] >= .84 else '尚未达到'}**。本地验证不等于官方公榜／私榜分数。", '',
              f"SHA-256：`{pointer['sha256']}`。旧版提交及模型保留。", '']
    lines += ['## 后续独立实验', '',
              '复制 `configs/` 中的配置，修改后用新名称运行，避免覆盖已完成结果：', '',
              '```sh', 'python3 arena_ood.py train --name my_new_trial --config configs/my_new_trial.json', '```', '',
              '主切分确定轮数后，再用 `--cutoff 2025-01-25 --fixed-epochs N` 固定轮数复核。历史 campaign 的 incumbent 和最终版本目录已经冻结，继续晋升时另建 campaign 和版本目录。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    readme = ROOT/'README.md'
    body = readme.read_text()
    if '<!-- PUSH084_START -->' in body:
        left, right = body.split('<!-- PUSH084_START -->', 1)
        _, tail = right.split('<!-- PUSH084_END -->', 1)
        body = left+tail.lstrip('\n')
    block = ['<!-- PUSH084_START -->', '# 继续向 0.84 优化', '',
             f"本轮完成 {len(selection['candidates'])} 组新配置及双时间切分复核。当前接受的最优本地 AUC：Jan26 **{pointer['validation_26']['auc']:.9f}**，Jan25 **{pointer['validation_25']['auc']:.9f}**。0.84 {'已达到（仅本地主验证）' if pointer['validation_26']['auc'] >= .84 else '尚未达到'}。", '',
             f"[可直接提交的最优 CSV]({pointer['csv']}) · [完整实验及 OOD 报告](competition/push084/REPORT.md) · [文件验收]({pointer['acceptance']})", '',
             '<!-- PUSH084_END -->', '']
    readme.write_text('\n'.join(block)+'\n'+body)
    print(json.dumps(dict(current_best=pointer, experiments=len(selection['candidates'])), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
