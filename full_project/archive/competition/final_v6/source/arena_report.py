"""Persist research, experiment evidence and the root acceptance report."""
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
LAB = ROOT / "competition"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temporary,path)


def note(message, kind="progress"):
    LAB.mkdir(exist_ok=True)
    with (LAB / "timeline.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), "kind": kind, "message": message}, ensure_ascii=False)+"\n")
    render()


def render():
    LAB.mkdir(exist_ok=True)
    results = []
    for path in sorted((LAB / "experiments").glob("*/result.json")):
        try:
            r = json.loads(path.read_text());r["path"] = str(path.relative_to(ROOT));results.append(r)
        except (ValueError, OSError):
            continue
    timeline = []
    starter_auc=next((r['auc'] for r in results if r['name']=='cf_26_fp64_audit'),0.615649)
    if (LAB / "timeline.jsonl").exists():
        timeline = [json.loads(line) for line in (LAB / "timeline.jsonl").read_text().splitlines() if line]
    manifest_path=LAB/'final/manifest.json';manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    acceptance_path=LAB/'final/acceptance.json';acceptance=json.loads(acceptance_path.read_text()) if acceptance_path.exists() else None
    import hashlib
    accepted=bool(manifest and acceptance and acceptance.get('passed') and acceptance.get('manifest_sha256')==hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    status='已完成训练、导出及独立验收' if accepted else '持续迭代中'
    lines = ["# TechArena 点击预测自动优化验收记录", "", f"更新时间：{now()}。所有时间为北京时间，数据切分日期为 UTC。", "",
             f"任务状态：{status}。以 ROC AUC 作为主指标，Logloss、平均精确率与 Accuracy 作为补充。评测集没有标签，不能在本机声称官方测试 AUC。", ""]
    v3_final=LAB/'final_v3'
    if (v3_final/'acceptance.json').exists():
        v3_manifest=json.loads((v3_final/'manifest.json').read_text());v3_acceptance=json.loads((v3_final/'acceptance.json').read_text())
        if v3_acceptance['passed'] and v3_acceptance['manifest_sha256']==hashlib.sha256((v3_final/'manifest.json').read_bytes()).hexdigest():
            lines += ['## 最新交付：V3 五轮 Transformer／基模迭代','',
                      f"完成 {v3_acceptance['training_iterations_completed']} 组新模型训练与双时间切点复核。最优配方 AUC：Jan26 **{v3_manifest['validation_26']['auc']:.9f}**，Jan25 **{v3_manifest['validation_25']['auc']:.9f}**。原 V2 与上一轮 Qwen3.5 LoRA 分支全部保留。",'',
                      '[本轮最优提交 CSV](competition/final_v3/submission.csv)，529,390 行；[完整五轮报告](competition/v3/REPORT.md)；[独立验收](competition/final_v3/acceptance.json)。',
                      '', '按双切点结果选择最终模型；若新尝试均未通过一致改进条件，交付文件使用冻结的既有最优预测。', '']
    if (LAB/'generative/incumbent.json').exists():
        lines += ['## Qwen3.5 生成式分类轻量分支','',
                  '独立入口为 `generative.py`。复用现有前向切分、112 维统计／匹配特征及三套 SGNS 行为表示，新增可训练向量投影、Qwen3.5-0.8B 语言骨干、最后四层 LoRA 和 A–K 单 token 标签输出行。K 表示不点击，最终取前十类概率。', '',
                  '每事件输入为用户／上下文 1 个向量位置、历史 20 个、候选 10 个、答案查询 1 个，共 32 个位置。仅计算最后位置的 11 个标签分数，避免为所有位置生成完整词表 logits。候选训练时随机重排并同步调整标签；历史保留时间顺序。', '',
                  '训练输入使用更早的 Jan24 表示快照，Jan26 验证输入使用截止 Jan26 的快照，并用过去共同商品做正交坐标对齐。当前最优模型和提交保留；此分支没有自动晋升或覆盖旧提交的入口。', '',
                  '```bash','python3 setup_generative.py',
                  'python3 generative_data.py --config configs/generative_smoke.json',
                  '.venv-generative/bin/python generative.py train --config configs/generative_smoke.json',
                  '.venv-generative/bin/python generative.py replay --run competition/generative/runs/qwen35_softtoken_smoke','```','',
                  '接近一轮的 1,000 事件试训使用 `configs/generative_light.json`（128 次更新，微批 4、梯度累积 2）；按上面命令替换配置，重载路径替换为 `competition/generative/runs/qwen35_softtoken_light`。', '',
                  '依赖安装在 `.venv-generative`，版本固定在 `requirements-generative.txt`；模型修订号与权重哈希见 `competition/generative/model_source.json`。模型使用 MLX／Metal 训练，已有 PyTorch 基线环境保留。', '',
                  '改训练量、更新步数、LoRA rank／层数：复制 `configs/generative_smoke.json` 并使用新 `name`；改输入：`generative_data.py`；改投影与骨干：`generative_model.py`；改优化、验证与重载：`generative.py`。训练量从现有 18 万事件特征集中抽取，独立验证随机种子保证训练量变化时验证事件不变。','']
        light_summary=LAB/'generative/runs/qwen35_softtoken_light/summary.json'
        summary_path=light_summary if light_summary.exists() else LAB/'generative/runs/qwen35_softtoken_smoke/summary.json'
        if summary_path.exists():
            g=json.loads(summary_path.read_text())
            lines += [f"真实轻量训练完成：准备 {g['train_events']:,} 个训练事件，执行 {g['steps']} 次更新，实际处理 {g['trained_event_presentations']} 次事件样本，约 {g['trained_event_presentations']/g['train_events']:.2f} 轮；固定验证 {g['valid_events']} 个事件。这属于小样本链路和资源测试，尚未完成大规模时间验证。",'',
                      '| 固定小验证队列 | AUC |','|---|---:|',
                      f"| Qwen3.5 微调前 | {g['initial_validation']['auc']:.6f} |",
                      f"| Qwen3.5 本次最佳更新 | {g['best_validation']['auc']:.6f} |",
                      f"| 当前 V2 同一队列参考 | {g['baseline_same_cohort']['auc']:.6f} |",'',
                      f"可训练参数 {g['trainable_parameters']:,}，峰值 MLX 内存 {g['peak_memory_gib']:.2f} GiB；常规每步中位耗时 {g['median_step_seconds']:.3f} 秒。适配器重载最大预测差 {g['adapter_replay_max_error']:.3g}。已有 V2 使用更多数据训练，这个小队列结果不能作为两种架构的公平优劣结论。",'',
                      f"固定训练探针交叉熵：{g['initial_train_probe_loss']:.4f} → {g['final_train_probe_loss']:.4f}。总耗时 {g['seconds']/60:.2f} 分钟。",'',
                      f"[完整测试结果]({summary_path.relative_to(ROOT)})。",'']
        lines += ['原始实现参考：[Qwen3.5 官方模型](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[MLX-LM Qwen3.5 向量输入接口](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py)、[LoRA 适配接口](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py)。','']
        trained=LAB/'generative/runs/qwen35_softtoken_8192_warm/summary.json'
        exported=LAB/'generative/final/qwen35_softtoken_8192_warm'
        if trained.exists():
            g=json.loads(trained.read_text())
            lines += ['### 扩大训练与完整提交','',
                      '本次从已训练的轻量检查点热启动，沿用其归一化参数，微批 8，AdamW 开启偏差修正。此前冷启动在第 256／512 步的验证 AUC 接近随机，已停止并保留在 `competition/generative/runs/qwen35_softtoken_8192/aborted.json`。','',
                      f"本机实际完成 {g['train_events']:,} 个训练事件、{g['steps']:,} 次更新、{g['trained_event_presentations']:,} 次事件样本；独立检查点选择队列 {g['valid_events']:,} 个事件，最佳 AUC {g['best_validation']['auc']:.6f}（第 {g['best_step']} 步）。训练耗时 {g['seconds']/60:.1f} 分钟，峰值 MLX 内存 {g['peak_memory_gib']:.2f} GiB。",'',
                      '[本轮训练结果](competition/generative/runs/qwen35_softtoken_8192_warm/summary.json)；[学习曲线](competition/generative/runs/qwen35_softtoken_8192_warm/training_curve.png)。','',
                      '这是受本机预算约束的抽样 LoRA 训练，并未用全部 686,824 个事件微调 LLM。SGNS 和统计表示仍使用各切点之前的可用数据。最终导出冻结验证训练所得头，使用 Jan27 快照并对齐到 Jan24；没有用 Jan26 标签再重训预测头。','',
                      '```bash',
                      'python3 generative_data.py --config configs/generative_warm.json',
                      '.venv-generative/bin/python generative.py train --config configs/generative_warm.json',
                      '.venv-generative/bin/python generative_export.py finish --run competition/generative/runs/qwen35_softtoken_8192_warm',
                      '```','',
                      '已完成的训练不会被覆盖。复用检查点只运行最后一条；新训练请先复制配置并更换 `name`。','']
        progress=trained.parent/'training.json'
        if not trained.exists() and progress.exists():
            records=json.loads(progress.read_text());checks=[r for r in records if 'validation' in r]
            current=f"，最近固定队列验证 AUC {checks[-1]['validation']['auc']:.6f}" if checks else ''
            lines += [f"扩大训练最新记录：已处理 {records[-1]['trained_event_presentations']:,}/8,192 次事件样本{current}。[训练日志](competition/generative/train_warm.log)。完整 CSV 在训练、完整时间验证和独立重载完成后生成。",'']
        if (exported/'acceptance.json').exists():
            gm=json.loads((exported/'manifest.json').read_text());ga=json.loads((exported/'acceptance.json').read_text())
            gv=gm['validation']
            lines += [f"生成式提交：[{gm['rows']:,} 行 CSV]({gm['csv']})；[独立验收]({(exported/'acceptance.json').relative_to(ROOT)})；[模型清单]({(exported/'manifest.json').relative_to(ROOT)})。验收通过：{ga['passed']}，新进程模型重载最大误差 {ga['fresh_process_model_replay_max_error']:.3g}。",'',
                      '| Jan26 时间验证队列 | Qwen LoRA AUC | 当前 V2 AUC |','|---|---:|---:|',
                      f"| 全部 {gv['full_time_validation']['events']:,} 个事件 | {gv['full_time_validation']['auc']:.6f} | {gv['incumbent_full_time_validation']['auc']:.6f} |",
                      f"| 排除选检查点队列的 {gv['heldout_time_validation']['events']:,} 个事件 | {gv['heldout_time_validation']['auc']:.6f} | {gv['incumbent_heldout_time_validation']['auc']:.6f} |",'',
                      '生成式 CSV 独立保存，未晋升为当前最优。正式冲分仍以通过双切点验收的 V2 提交为准。','']
            diagnostic=exported/'blend_diagnostic.json'
            if diagnostic.exists():
                chosen=json.loads(diagnostic.read_text())['selected_on_2048']
                lines += [f"另检验了概率和 Logit 小权重融合，固定选择队列选出的 LLM 权重为 {chosen['llm_weight']:g}。[融合诊断]({diagnostic.relative_to(ROOT)})。",'']
    iteration_path=LAB/'iteration_v2.json'
    if iteration_path.exists():
        iteration=json.loads(iteration_path.read_text())
        lines += ['## 第二版结构扩展','',f"状态：{iteration['status']}。v1 最优提交、清单和验收已冻结在 [champions/v1](competition/champions/v1/snapshot.json)，完整模型版本保留。",
                  '新结构入口 `arena_neural_v2.py`：更宽的 MLP、DCN V2 式矩阵交叉、候选间自注意力；同一特征与事件级损失进行比较。扩大网络本身不保证提升，只有通过时间复核和独立验收才晋升推荐版本。','']
        if iteration.get('report'):lines += [f"本轮结果：[报告]({iteration['report']})。",'']
        v2_path=LAB/'v2_selection.json'
        if v2_path.exists():
            v2=json.loads(v2_path.read_text());old=v2['incumbent'];new=v2['selected']
            lines += ['| 配方 | Jan26 AUC | Jan25 AUC |','|---|---:|---:|',
                      f"| 冻结 v1 | {old['auc_26']:.6f} | {old['auc_25']:.6f} |",
                      f"| 本轮选定配方 | {new['auc_26']:.6f} | {new['auc_25']:.6f} |",'',
                      '| 新神经头 | 评分网络参数 | Jan26 AUC | Jan25 锁定复核 AUC |',
                      '|---|---:|---:|---:|']
            audits={r['name']:r['auc'] for r in v2['secondary_results']}
            for r in v2['primary_results']:
                if r['name'].startswith('v2_'):
                    lines.append(f"| {r['name']} | {r['parameters']:,} | {r['auc']:.6f} | {audits[r['name']+'_locked25']:.6f} |")
            lines += ['', '原神经评分头为 33,730 参数。多表示版本额外输入另一套独立 SGNS 表示的 12 个匹配特征；仍然只使用对应时间点之前的数据。', '',
                      '继续或复现本轮（自动保留已完成实验）：','', '```bash','python3 arena_iterate_v2.py','```','',
                      '所有网络配置位于 `competition/experiments/v2_*/config.json`。改结构使用 `arena_neural_v2.py`，改 `hidden`、`cross_layers`、`attention_layers` 或 `multi_view` 时使用新实验名。', '']
            comparison_path=LAB/'v2_comparison.json'
            if comparison_path.exists():
                comparisons=json.loads(comparison_path.read_text())
                lines += ['同一事件配对比较（固定已选配方，区间不包含选模型的不确定性）：','',
                          '| 时间切点 | v1 AUC | v2 AUC | AUC 差值 | 差值 95% 区间 |',
                          '|---|---:|---:|---:|---|']
                for day,r in comparisons['cutoffs'].items():
                    lo,hi=r['gain_vs_v1_95_ci']
                    lines.append(f"| {day} | {r['v1_auc']:.6f} | {r['v2_auc']:.6f} | {r['gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
                lines += ['', '完整结果：[v2_comparison.json](competition/v2_comparison.json)。', '']
    if manifest:
        lines += ['## 直接查看交付','',f"- 提交文件：[submission.csv]({manifest['submission']})，共 {manifest['rows']:,} 行。",
                  f"- 模型、表示与代码快照：[{manifest['model_dir']}]({manifest['model_dir']})。",
                  f"- 双时间切点 AUC：Jan 26 **{manifest['validation_26']['auc']:.6f}**；Jan 25 **{manifest['validation_25']['auc']:.6f}**。",
                  '- 配方与哈希：[manifest.json](competition/final/manifest.json)。']
        if accepted:
            lines += ['- 独立验收：[acceptance.json](competition/final/acceptance.json)，包含模型重载、特征重算、CSV 检查和复现误差。',
                      f"- 官方 starter 公式双精度复核同一 Jan 26 验证 AUC {starter_auc:.6f}；最终提升 **{manifest['validation_26']['auc']-starter_auc:.6f}**。",
                      f"- 已保存模型重新预测的最大概率差：{acceptance['reproduction_max_probability_error']:.3g}。"]
        if 'validation_24' in manifest:
            lines += [f"- 未参与配方选择的 Jan 24 回溯复核 AUC：**{manifest['validation_24']['auc']:.6f}**。"]
        lines += ['', '最终方案先从历史点击学习小维度用户／商品表示，再结合严格前向的统计、匹配和上下文特征训练评分头，最后按两个时间切点选择简单融合。具体成员、权重与损失见清单。', '']
        primary=[r for r in results if r.get('cutoff')=='2025-01-26']
        lines += ['同一 Jan 26 验证集上的迭代路径（各类已完成实验的最好结果，并非单因素消融）：','',
                  '| 阶段 | AUC |','|---|---:|','| 原历史双塔 | 0.585356 |',f'| 官方 starter 公式 FP64 | {starter_auc:.6f} |']
        for label,methods in [('隐式 ALS',('implicit_als',)),('采样反馈低维表示',('sgns_mps',)),
                              ('表示与统计特征的树评分头',('catboost_numeric',)),('事件级神经评分头',('tabular_mlp_mps',))]:
            values=[r['auc'] for r in primary if r['method'] in methods]
            if values:lines.append(f'| {label} | {max(values):.6f} |')
        lines += [f"| 最终冻结融合 | {manifest['validation_26']['auc']:.6f} |",'']
    lines += ['## 轻量链路与后续实验','',
              '轻量版本已完成 baseline → 少量训练 → 保存加载 → 完整评测预测 → CSV 检查。详见 [LIGHT_START.md](LIGHT_START.md)。',
              '', '| 轻量实验阶段 | 固定小验证集 AUC |','|---|---:|','| 平滑用户／商品 CTR | 0.550995 |','| 两轮 8 维历史双塔，最佳轮 | 0.581668 |','',
              '```bash','python3 light.py','python3 light.py --name history5 --history-limit 5 --skip-predict',
              'python3 light.py --name dim16 --dim 16 --skip-predict','```','',
              '该小验证集与下面完整时间验证集不同，分数不能横向比较。数据抽样和训练量修改 `light_data.py`，历史处理修改 `ctr_data.py`，双塔修改 `ctr_model.py`，参数在 `configs/light.json`。', '',
             "## 运行与资源", "", "本机 Apple M2，8 核 CPU，16 GB 统一内存，MPS GPU 可用。只使用本机算力；依赖与 Go 工具链安装在 `.tools/`。", "",
             "入口为 `overnight.go`：调度检索实验记忆、生成候选配置、训练验证、更新记录、最终导出与独立验收。Go 负责任务调度，数值训练由 Python/原生计算库执行。", "",
             '```bash','.tools/go/bin/go run overnight.go -max-experiments 30 -hours 4.5',
             '# 只使用已保存的最终模型重新预测','python3 arena_finalize.py --predict-only',
             '# 重新执行独立验收','python3 arena_verify.py','```','',
             '依赖版本见最终版本目录中的 `environment.json`。CPU 通道使用最多 6 线程，GPU 使用 MPS；避免重复启动相同通道，已有存活任务会先等待。资源采样见 `competition/resources.jsonl`。', '',
             "## 验证约束", "", "每次均按全局时间切分，取切点后每个用户的首个完整事件验证。训练特征统计和协同过滤只使用更早的拟合数据。树模型训练行使用更早快照生成特征，避免标签进入自身统计。", "",
             'Jan 26 用于主实验及早停；锁定模型参数后在 Jan 25 滚动复核。融合比较预设的粗权重，并使用两切点按评测集无标签历史长度／冷启动分布加权的平均诊断 AUC 选择；原始全局 AUC 同时记录。Jan 24 最后冻结复核，不参与配方选择。置信区间对固定配方按完整事件重采样，不包括模型选择的不确定性。', '',
             "历史是已发生点击，可作为辅助隐式反馈；去重用户物品交互，不把同一历史在十个候选和后续事件中的重复当成独立标签。保留未点击事件和冷启动用户。", "",
             "既有完整切分结果：2025-01-26 的平滑 CTR AUC 0.546477，历史双塔 AUC 0.585356；轻量实验用另一小验证集，不能直接与完整验证分数比较。", "",
             "## 实验结果", "", "| 实验 | 切分 | AUC | Logloss | 耗时秒 | 记录 |", "|---|---|---:|---:|---:|---|"]
    for r in sorted(results, key=lambda x: (x.get("cutoff", ""), -x.get("auc", 0))):
        lines.append(f"| {r['name']} | {r.get('cutoff','')} | {r.get('auc',0):.6f} | {r.get('logloss',0):.6f} | {r.get('seconds',0):.1f} | [{r['name']}]({r['path']}) |")
    if not results:
        lines.append("| 新一轮实验正在准备 | — | — | — | — | — |")
    final = LAB / "final" / "manifest.json"
    lines += ["", "## 最终交付", ""]
    if final.exists():
        manifest = json.loads(final.read_text())
        lines += [f"- 推荐提交：[{manifest['submission']}]({manifest['submission']})",
                  f"- 模型与复现配置：[{manifest['model_dir']}]({manifest['model_dir']})",
                  "", "```json", json.dumps(manifest, indent=2, ensure_ascii=False), "```"]
    else:
        lines += ["最终目录预留为 `competition/final/`。当前已有的初版提交在 `submissions/two_tower_v1.csv`，新一轮完成后将给出更强模型及校验清单。"]
    lines += ["", "## 检索增强的自迭代", "", "检索对象是论文与官方文档摘要、数据诊断和本地实验记录。配置生成器检索相近问题，结合已有指标选择下一组参数；检索内容与理由保存到每个实验目录。推荐模型自身负责数值点击预测，避免将不必要的语言模型推理加入每条样本。", "",
              "研究资料保存在 `competition/research/`；参考原始论文与官方文档，实验优劣以本地时间验证为准。配置生成使用本地 TF-IDF 检索与基于指标的规则；未调用付费语言模型 API。", "",
              '主要原始资料：[时间泄漏研究](https://arxiv.org/abs/2010.11060)、[implicit ALS 文档](https://benfred.github.io/implicit/api/models/cpu/als.html)、[Item2Vec](https://arxiv.org/abs/1603.04259)、[DIN](https://arxiv.org/abs/1706.06978)、[LightGCN](https://arxiv.org/abs/2002.02126)。具体借鉴及未采用原因见每轮 `retrieval_plan.json` 与执行记录。', '',
              "## 全部执行与取舍记录", ""]
    lines += [f"- {r['time']} [{r['kind']}] {r['message']}" for r in timeline]
    lines += ["", "## 目录", "", "- `competition/experiments/`：每轮配置、指标、验证预测与模型。", "- `competition/cache/`：只按拟合时间构建的稀疏交互、协同矩阵和特征缓存。", "- `competition/research/`：资料、检索结果和实验计划。", "- `competition/logs/`：Go 调度与训练原始日志。", "- `competition/final/`：最终推荐 CSV、模型清单、校验与复现入口。", "- `LIGHT_START.md`：之前的轻量链路教程。", ""]
    lines += ['全量流程的改造入口：`arena_features.py` 汇总特征，`arena_feedback.py` 处理历史反馈与上下文统计，`arena_sgns.py` 训练辅助表示，`arena_tabular.py` 定义神经评分头，`arena_plan.py` 定义检索和候选实验。每次用新实验名保留配置与结果；只修改 SGNS 训练轮数也会校验缓存配方，避免复用不匹配的特征。','']
    target=ROOT/"README.md"
    temp=target.with_name(f"README.md.{os.getpid()}.tmp")
    temp.write_text("\n".join(lines),encoding="utf-8")
    os.replace(temp,target)


if __name__ == "__main__":
    render()
