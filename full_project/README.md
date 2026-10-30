<!-- CURRENT_STATUS_START -->
**当前可提交最优：V31** · [submission.csv](competition/final_v31/submission.csv) · [本轮三次迭代与边缘样本报告](competition/iteration_29_31/REPORT.md)

**独立复现包：[solution_v31/README.md](solution_v31/README.md)**。包含本次 evaluation、必要推理权重和固定 V30 上游预测；复制整个文件夹即可复现 V31 增量推理，并导出与已验收提交逐字节一致的 CSV。

| 本地验证切分 | AUC |
|---|---:|
| Jan26 | 0.832004960 |
| Jan25 | 0.822609304 |
| Jan24 | 0.808895070 |

目标 **0.838 尚未达到**。529,390行和冻结源码独立复算通过，误差0；所有旧提交保留。近分事件AUC相对V28在三切分均升，首位命中率仍有一个切分回退。官方分数未知。

<!-- CURRENT_STATUS_END -->

<!-- PUSH084_V31_AFTER_V30_START -->
# 继续向 0.838 优化

新增 2 组配置；晋升：True。当前最优本地 Jan26 AUC **0.832004960**，Jan25 **0.822609304**；0.838 尚未达到。

[当前最优 CSV](competition/final_v31/submission.csv) · [本轮报告](competition/push084_v31_after_v30/REPORT.md)

<!-- PUSH084_V31_AFTER_V30_END -->

<!-- LATENT_DELTA_START -->
# 最新候选偏好修正

晋升：True；当前最优本地主切分 AUC **0.831985479**。目标 0.838 尚未达到。

[直接提交 CSV](competition/final_v30/submission.csv) · [本轮报告](competition/third_sphere_final/REPORT.md)

<!-- LATENT_DELTA_END -->

<!-- CONDITIONAL_SPHERE_START -->
本地最优 AUC **0.831978378**；目标 **0.838 尚未达到**。

[可提交 CSV](competition/final_v29/submission.csv) · [候选偏好修正报告](competition/conditional_sphere/REPORT.md) · [历史注意力对照](competition/history_rerank/REPORT.md)
<!-- CONDITIONAL_SPHERE_END -->

<!-- EVENT_MASS_START -->
当前可提交最优：[competition/final_v28/submission.csv](competition/final_v28/submission.csv)。本地 Jan26 AUC **0.831858926**；目标 **0.838 尚未达到**。

[事件点击概率修正报告](competition/event_mass/REPORT.md) · [真实曝光条件重排](competition/conditional_sphere/REPORT.md) · [近分前两名对照](competition/borderline_pair/REPORT.md)
<!-- EVENT_MASS_END -->

<!-- PUSH084_V25_START -->
# 继续向 0.838 优化

新增 4 组配置；晋升：False。当前最优本地 Jan26 AUC **0.831775451**，Jan25 **0.821976953**；0.838 尚未达到。

[当前最优 CSV](competition/final_v24/submission.csv) · [本轮报告](competition/push084_v25/REPORT.md)

<!-- PUSH084_V25_END -->

<!-- PUSH084_V24_START -->
# 继续向 0.838 优化

新增 4 组配置；晋升：True。当前最优本地 Jan26 AUC **0.831775451**，Jan25 **0.821976953**；0.838 尚未达到。

[当前最优 CSV](competition/final_v24/submission.csv) · [本轮报告](competition/push084_v24/REPORT.md)

<!-- PUSH084_V24_END -->

<!-- METRIC_OOF_START -->
# 最新时间样本外匹配修正

新增 6 组小参数匹配实验；晋升：True。当前最优本地主切分 AUC **0.831733752**；目标 **0.838 尚未达到**。

[最优提交 CSV](competition/final_v23/submission.csv) · [本轮报告](competition/metric_oof/REPORT.md)

<!-- METRIC_OOF_END -->

<!-- TWO_STAGE_START -->
# 当前最优 V21：时间样本外精排＋重排，目标 0.838

本轮完成 7 组神经重排配置与 4 组用户偏好后验探针。V21 已通过独立进程完整原始输入重算，误差 0，提交 529,390 行；旧版本均保留。**0.838 尚未达到。**

| 时间切分 | V18 | V21 |
|---|---:|---:|
| Jan26 | 0.831664252 | 0.831712062 |
| Jan25 | 0.821762817 | 0.821835164 |
| Jan24 | 0.808095222 | 0.808240288 |

单个精排模型生成逐日时间样本外分数；第二轮 Transformer 使用 244 维特征、候选相似度、置信度门控和全局排序损失。将学到的效用修正以 50% 幅度迁移到 V18。后续首次查询／独立候选选择／个人偏好后验实验未超越 V21。

主切分重采样提升区间包含 0，边缘样本提升很小；以上为本地分数，官方评测标签不可用。Jan24/25/26 部分查询重叠，不是统计独立的三个验证集。

[直接提交 CSV](competition/final_v21/submission.csv) · [两轮方案实验](competition/two_stage/REPORT.md) · [首次查询实验](competition/two_stage_origins/REPORT.md) · [个人偏好探针](competition/oof_user_rerank/REPORT.md)

<!-- TWO_STAGE_END -->

<!-- PUSH084_V19_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：False。当前最优本地 Jan26 AUC **0.831664252**，Jan25 **0.821762817**；0.84 尚未达到。

[当前最优 CSV](competition/final_v18/submission.csv) · [本轮报告](competition/push084_v19/REPORT.md)

<!-- PUSH084_V19_END -->

<!-- PUSH084_V18_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：True。当前最优本地 Jan26 AUC **0.831664252**，Jan25 **0.821762817**；0.84 尚未达到。

[当前最优 CSV](competition/final_v18/submission.csv) · [本轮报告](competition/push084_v18/REPORT.md)

<!-- PUSH084_V18_END -->

<!-- PUSH084_V17_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：False。当前最优本地 Jan26 AUC **0.831620522**，Jan25 **0.821722732**；0.84 尚未达到。

[当前最优 CSV](competition/final_v16/submission.csv) · [本轮报告](competition/push084_v17/REPORT.md)

<!-- PUSH084_V17_END -->

<!-- PUSH084_V16_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：True。当前最优本地 Jan26 AUC **0.831620522**，Jan25 **0.821722732**；0.84 尚未达到。

[当前最优 CSV](competition/final_v16/submission.csv) · [本轮报告](competition/push084_v16/REPORT.md)

<!-- PUSH084_V16_END -->

<!-- PUSH084_V15_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：True。当前最优本地 Jan26 AUC **0.831602473**，Jan25 **0.821705748**；0.84 尚未达到。

[当前最优 CSV](competition/final_v15/submission.csv) · [本轮报告](competition/push084_v15/REPORT.md)

<!-- PUSH084_V15_END -->

<!-- PUSH084_V14_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：True。当前最优本地 Jan26 AUC **0.831550289**，Jan25 **0.821691533**；0.84 尚未达到。

[当前最优 CSV](competition/final_v14/submission.csv) · [本轮报告](competition/push084_v14/REPORT.md)

<!-- PUSH084_V14_END -->

<!-- PUSH084_V13_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：False。当前最优本地 Jan26 AUC **0.831508248**，Jan25 **0.821678395**；0.84 尚未达到。

[当前最优 CSV](competition/final_v12/submission.csv) · [本轮报告](competition/push084_v13/REPORT.md)

<!-- PUSH084_V13_END -->

<!-- PUSH084_V12_START -->
# 继续向 0.84 优化

新增 5 组配置；晋升：True。当前最优本地 Jan26 AUC **0.831508248**，Jan25 **0.821678395**；0.84 尚未达到。

[当前最优 CSV](competition/final_v12/submission.csv) · [本轮报告](competition/push084_v12/REPORT.md)

<!-- PUSH084_V12_END -->

<!-- PUSH084_V11_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：True。当前最优本地 Jan26 AUC **0.831419164**，Jan25 **0.821484065**；0.84 尚未达到。

[当前最优 CSV](competition/final_v11/submission.csv) · [本轮报告](competition/push084_v11/REPORT.md)

<!-- PUSH084_V11_END -->

<!-- PUSH084_V10_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：False。当前最优本地 Jan26 AUC **0.831399146**，Jan25 **0.821470227**；0.84 尚未达到。

[当前最优 CSV](competition/final_v9/submission.csv) · [本轮报告](competition/push084_v10/REPORT.md)

<!-- PUSH084_V10_END -->

<!-- PUSH084_V9_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：True。当前最优本地 Jan26 AUC **0.831399146**，Jan25 **0.821470227**；0.84 尚未达到。

[当前最优 CSV](competition/final_v9/submission.csv) · [本轮报告](competition/push084_v9/REPORT.md)

<!-- PUSH084_V9_END -->

<!-- PUSH084_V8_START -->
# 继续向 0.84 优化

新增 4 组配置；晋升：True。当前最优本地 Jan26 AUC **0.831382749**，Jan25 **0.821414586**；0.84 尚未达到。

[当前最优 CSV](competition/final_v8/submission.csv) · [本轮报告](competition/push084_v8/REPORT.md)

<!-- PUSH084_V8_END -->

<!-- PUSH084_START -->
# 继续向 0.84 优化

V6 历史迭代：完成 23 组新配置及双时间切分复核；该轮 AUC 为 Jan26 **0.831314174**、Jan25 **0.821243980**。最新接受版本见文首。

[保留的 V6 CSV](competition/final_v6/submission.csv) · [完整实验及 OOD 报告](competition/push084/REPORT.md) · [文件验收](competition/final_v6/acceptance.json)

<!-- PUSH084_END -->

# TechArena 点击预测自动优化验收记录

更新时间：2026-10-25T06:06:38+08:00。所有时间为北京时间，数据切分日期为 UTC。

任务状态：已完成训练、导出及独立验收。以 ROC AUC 作为主指标，Logloss、平均精确率与 Accuracy 作为补充。评测集没有标签，不能在本机声称官方测试 AUC。

## 最新交付：V3 五轮 Transformer／基模迭代

完成 5 组新模型训练与双时间切点复核。最优配方 AUC：Jan26 **0.829341153**，Jan25 **0.819250413**。原 V2 与上一轮 Qwen3.5 LoRA 分支全部保留。

[本轮最优提交 CSV](competition/final_v3/submission.csv)，529,390 行；[完整五轮报告](competition/v3/REPORT.md)；[独立验收](competition/final_v3/acceptance.json)。

按双切点结果选择最终模型；若新尝试均未通过一致改进条件，交付文件使用冻结的既有最优预测。

## Qwen3.5 生成式分类轻量分支

独立入口为 `generative.py`。复用现有前向切分、112 维统计／匹配特征及三套 SGNS 行为表示，新增可训练向量投影、Qwen3.5-0.8B 语言骨干、最后四层 LoRA 和 A–K 单 token 标签输出行。K 表示不点击，最终取前十类概率。

每事件输入为用户／上下文 1 个向量位置、历史 20 个、候选 10 个、答案查询 1 个，共 32 个位置。仅计算最后位置的 11 个标签分数，避免为所有位置生成完整词表 logits。候选训练时随机重排并同步调整标签；历史保留时间顺序。

训练输入使用更早的 Jan24 表示快照，Jan26 验证输入使用截止 Jan26 的快照，并用过去共同商品做正交坐标对齐。当前最优模型和提交保留；此分支没有自动晋升或覆盖旧提交的入口。

```bash
python3 setup_generative.py
python3 generative_data.py --config configs/generative_smoke.json
.venv-generative/bin/python generative.py train --config configs/generative_smoke.json
.venv-generative/bin/python generative.py replay --run competition/generative/runs/qwen35_softtoken_smoke
```

接近一轮的 1,000 事件试训使用 `configs/generative_light.json`（128 次更新，微批 4、梯度累积 2）；按上面命令替换配置，重载路径替换为 `competition/generative/runs/qwen35_softtoken_light`。

依赖安装在 `.venv-generative`，版本固定在 `requirements-generative.txt`；模型修订号与权重哈希见 `competition/generative/model_source.json`。模型使用 MLX／Metal 训练，已有 PyTorch 基线环境保留。

改训练量、更新步数、LoRA rank／层数：复制 `configs/generative_smoke.json` 并使用新 `name`；改输入：`generative_data.py`；改投影与骨干：`generative_model.py`；改优化、验证与重载：`generative.py`。训练量从现有 18 万事件特征集中抽取，独立验证随机种子保证训练量变化时验证事件不变。

真实轻量训练完成：准备 1,000 个训练事件，执行 128 次更新，实际处理 1024 次事件样本，约 1.02 轮；固定验证 128 个事件。这属于小样本链路和资源测试，尚未完成大规模时间验证。

| 固定小验证队列 | AUC |
|---|---:|
| Qwen3.5 微调前 | 0.450121 |
| Qwen3.5 本次最佳更新 | 0.618767 |
| 当前 V2 同一队列参考 | 0.830521 |

可训练参数 1,487,627，峰值 MLX 内存 6.79 GiB；常规每步中位耗时 3.936 秒。适配器重载最大预测差 0。已有 V2 使用更多数据训练，这个小队列结果不能作为两种架构的公平优劣结论。

固定训练探针交叉熵：1.7034 → 1.5449。总耗时 9.18 分钟。

[完整测试结果](competition/generative/runs/qwen35_softtoken_light/summary.json)。

原始实现参考：[Qwen3.5 官方模型](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[MLX-LM Qwen3.5 向量输入接口](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py)、[LoRA 适配接口](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py)。

### 扩大训练与完整提交

本次从已训练的轻量检查点热启动，沿用其归一化参数，微批 8，AdamW 开启偏差修正。此前冷启动在第 256／512 步的验证 AUC 接近随机，已停止并保留在 `competition/generative/runs/qwen35_softtoken_8192/aborted.json`。

本机实际完成 8,192 个训练事件、1,024 次更新、8,192 次事件样本；独立检查点选择队列 2,048 个事件，最佳 AUC 0.756904（第 1024 步）。训练耗时 58.7 分钟，峰值 MLX 内存 5.85 GiB。

[本轮训练结果](competition/generative/runs/qwen35_softtoken_8192_warm/summary.json)；[学习曲线](competition/generative/runs/qwen35_softtoken_8192_warm/training_curve.png)。

这是受本机预算约束的抽样 LoRA 训练，并未用全部 686,824 个事件微调 LLM。SGNS 和统计表示仍使用各切点之前的可用数据。最终导出冻结验证训练所得头，使用 Jan27 快照并对齐到 Jan24；没有用 Jan26 标签再重训预测头。

```bash
python3 generative_data.py --config configs/generative_warm.json
.venv-generative/bin/python generative.py train --config configs/generative_warm.json
.venv-generative/bin/python generative_export.py finish --run competition/generative/runs/qwen35_softtoken_8192_warm
```

已完成的训练不会被覆盖。复用检查点只运行最后一条；新训练请先复制配置并更换 `name`。

生成式提交：[529,390 行 CSV](competition/generative/final/qwen35_softtoken_8192_warm/submission.csv)；[独立验收](competition/generative/final/qwen35_softtoken_8192_warm/acceptance.json)；[模型清单](competition/generative/final/qwen35_softtoken_8192_warm/manifest.json)。验收通过：True，新进程模型重载最大误差 0。

| Jan26 时间验证队列 | Qwen LoRA AUC | 当前 V2 AUC |
|---|---:|---:|
| 全部 35,138 个事件 | 0.751811 | 0.829327 |
| 排除选检查点队列的 32,975 个事件 | 0.751569 | 0.828960 |

生成式 CSV 独立保存，未晋升为当前最优。正式冲分仍以通过双切点验收的 V2 提交为准。

另检验了概率和 Logit 小权重融合，固定选择队列选出的 LLM 权重为 0。[融合诊断](competition/generative/final/qwen35_softtoken_8192_warm/blend_diagnostic.json)。

## 第二版结构扩展

状态：completed; v2 promoted。v1 最优提交、清单和验收已冻结在 [champions/v1](competition/champions/v1/snapshot.json)，完整模型版本保留。
新结构入口 `arena_neural_v2.py`：更宽的 MLP、DCN V2 式矩阵交叉、候选间自注意力；同一特征与事件级损失进行比较。扩大网络本身不保证提升，只有通过时间复核和独立验收才晋升推荐版本。

本轮结果：[报告](competition/v2_selection.json)。

| 配方 | Jan26 AUC | Jan25 AUC |
|---|---:|---:|
| 冻结 v1 | 0.829004 | 0.818804 |
| 本轮选定配方 | 0.829327 | 0.819246 |

| 新神经头 | 评分网络参数 | Jan26 AUC | Jan25 锁定复核 AUC |
|---|---:|---:|---:|
| v2_wide256_s2026 | 113,922 | 0.827959 | 0.816084 |
| v2_cross256_s2026 | 185,070 | 0.827848 | 0.815776 |
| v2_attention128_s2026 | 442,786 | 0.827647 | 0.816357 |
| v2_multiview_wide256_s2026 | 177,058 | 0.829246 | 0.818649 |
| v2_multiview_cross256_s2026 | 202,482 | 0.829103 | 0.818431 |

原神经评分头为 33,730 参数。多表示版本额外输入另一套独立 SGNS 表示的 12 个匹配特征；仍然只使用对应时间点之前的数据。

继续或复现本轮（自动保留已完成实验）：

```bash
python3 arena_iterate_v2.py
```

所有网络配置位于 `competition/experiments/v2_*/config.json`。改结构使用 `arena_neural_v2.py`，改 `hidden`、`cross_layers`、`attention_layers` 或 `multi_view` 时使用新实验名。

同一事件配对比较（固定已选配方，区间不包含选模型的不确定性）：

| 时间切点 | v1 AUC | v2 AUC | AUC 差值 | 差值 95% 区间 |
|---|---:|---:|---:|---|
| 2025-01-26 | 0.829004 | 0.829327 | +0.000323 | [+0.000217, +0.000421] |
| 2025-01-25 | 0.818804 | 0.819246 | +0.000442 | [+0.000273, +0.000587] |
| 2025-01-24 | 0.804915 | 0.805402 | +0.000487 | [+0.000356, +0.000616] |

完整结果：[v2_comparison.json](competition/v2_comparison.json)。

## 直接查看交付

- 提交文件：[submission.csv](competition/final_v2/submission.csv)，共 529,390 行。
- 模型、表示与代码快照：[competition/final_v2/versions/20261024_110038](competition/final_v2/versions/20261024_110038)。
- 双时间切点 AUC：Jan 26 **0.829327**；Jan 25 **0.819246**。
- 配方与哈希：[manifest.json](competition/final/manifest.json)。
- 独立验收：[acceptance.json](competition/final/acceptance.json)，包含模型重载、特征重算、CSV 检查和复现误差。
- 官方 starter 公式双精度复核同一 Jan 26 验证 AUC 0.615704；最终提升 **0.213624**。
- 已保存模型重新预测的最大概率差：0。
- 未参与配方选择的 Jan 24 回溯复核 AUC：**0.805402**。

最终方案先从历史点击学习小维度用户／商品表示，再结合严格前向的统计、匹配和上下文特征训练评分头，最后按两个时间切点选择简单融合。具体成员、权重与损失见清单。

同一 Jan 26 验证集上的迭代路径（各类已完成实验的最好结果，并非单因素消融）：

| 阶段 | AUC |
|---|---:|
| 原历史双塔 | 0.585356 |
| 官方 starter 公式 FP64 | 0.615704 |
| 隐式 ALS | 0.677857 |
| 采样反馈低维表示 | 0.803856 |
| 表示与统计特征的树评分头 | 0.825058 |
| 事件级神经评分头 | 0.829267 |
| 最终冻结融合 | 0.829327 |

## 轻量链路与后续实验

轻量版本已完成 baseline → 少量训练 → 保存加载 → 完整评测预测 → CSV 检查。详见 [LIGHT_START.md](LIGHT_START.md)。

| 轻量实验阶段 | 固定小验证集 AUC |
|---|---:|
| 平滑用户／商品 CTR | 0.550995 |
| 两轮 8 维历史双塔，最佳轮 | 0.581668 |

```bash
python3 light.py
python3 light.py --name history5 --history-limit 5 --skip-predict
python3 light.py --name dim16 --dim 16 --skip-predict
```

该小验证集与下面完整时间验证集不同，分数不能横向比较。数据抽样和训练量修改 `light_data.py`，历史处理修改 `ctr_data.py`，双塔修改 `ctr_model.py`，参数在 `configs/light.json`。

## 运行与资源

本机 Apple M2，8 核 CPU，16 GB 统一内存，MPS GPU 可用。只使用本机算力；依赖与 Go 工具链安装在 `.tools/`。

入口为 `overnight.go`：调度检索实验记忆、生成候选配置、训练验证、更新记录、最终导出与独立验收。Go 负责任务调度，数值训练由 Python/原生计算库执行。

```bash
.tools/go/bin/go run overnight.go -max-experiments 30 -hours 4.5
# 只使用已保存的最终模型重新预测
python3 arena_finalize.py --predict-only
# 重新执行独立验收
python3 arena_verify.py
```

依赖版本见最终版本目录中的 `environment.json`。CPU 通道使用最多 6 线程，GPU 使用 MPS；避免重复启动相同通道，已有存活任务会先等待。资源采样见 `competition/resources.jsonl`。

## 验证约束

每次均按全局时间切分，取切点后每个用户的首个完整事件验证。训练特征统计和协同过滤只使用更早的拟合数据。树模型训练行使用更早快照生成特征，避免标签进入自身统计。

Jan 26 用于主实验及早停；锁定模型参数后在 Jan 25 滚动复核。融合比较预设的粗权重，并使用两切点按评测集无标签历史长度／冷启动分布加权的平均诊断 AUC 选择；原始全局 AUC 同时记录。Jan 24 最后冻结复核，不参与配方选择。置信区间对固定配方按完整事件重采样，不包括模型选择的不确定性。

历史是已发生点击，可作为辅助隐式反馈；去重用户物品交互，不把同一历史在十个候选和后续事件中的重复当成独立标签。保留未点击事件和冷启动用户。

既有完整切分结果：2025-01-26 的平滑 CTR AUC 0.546477，历史双塔 AUC 0.585356；轻量实验用另一小验证集，不能直接与完整验证分数比较。

## 实验结果

| 实验 | 切分 | AUC | Logloss | 耗时秒 | 记录 |
|---|---|---:|---:|---:|---|
| v2_multiview_wide256_s2026_locked24 | 2025-01-24 | 0.805364 | 0.195426 | 32.9 | [v2_multiview_wide256_s2026_locked24](competition/experiments/v2_multiview_wide256_s2026_locked24/result.json) |
| v2_multiview_cross256_s2026_locked24 | 2025-01-24 | 0.804750 | 0.195683 | 21.6 | [v2_multiview_cross256_s2026_locked24](competition/experiments/v2_multiview_cross256_s2026_locked24/result.json) |
| event_cat_sgns8e640_context24_d6_locked24 | 2025-01-24 | 0.802905 | 0.195861 | 9.7 | [event_cat_sgns8e640_context24_d6_locked24](competition/experiments/event_cat_sgns8e640_context24_d6_locked24/result.json) |
| mlp_event_cat_sgns8_s2026_context24_d6_locked24 | 2025-01-24 | 0.802686 | 0.196394 | 28.7 | [mlp_event_cat_sgns8_s2026_context24_d6_locked24](competition/experiments/mlp_event_cat_sgns8_s2026_context24_d6_locked24/result.json) |
| cat_sgns8e640_context24_d6_locked24 | 2025-01-24 | 0.800866 | 0.196933 | 401.2 | [cat_sgns8e640_context24_d6_locked24](competition/experiments/cat_sgns8e640_context24_d6_locked24/result.json) |
| v3_wide512_s73_locked25 | 2025-01-25 | 0.818701 | 0.192254 | 50.6 | [v3_wide512_s73_locked25](competition/experiments/v3_wide512_s73_locked25/result.json) |
| v2_multiview_wide256_s2026_locked25 | 2025-01-25 | 0.818649 | 0.192281 | 30.1 | [v2_multiview_wide256_s2026_locked25](competition/experiments/v2_multiview_wide256_s2026_locked25/result.json) |
| v3_cross512_s2027_locked25 | 2025-01-25 | 0.818591 | 0.192167 | 29.8 | [v3_cross512_s2027_locked25](competition/experiments/v3_cross512_s2027_locked25/result.json) |
| v2_multiview_cross256_s2026_locked25 | 2025-01-25 | 0.818431 | 0.192558 | 20.8 | [v2_multiview_cross256_s2026_locked25](competition/experiments/v2_multiview_cross256_s2026_locked25/result.json) |
| v2_attention128_s2026_locked25 | 2025-01-25 | 0.816357 | 0.192963 | 80.8 | [v2_attention128_s2026_locked25](competition/experiments/v2_attention128_s2026_locked25/result.json) |
| mlp_event_cat_sgns8e640_context24_d6_locked25 | 2025-01-25 | 0.816275 | 0.193287 | 28.9 | [mlp_event_cat_sgns8e640_context24_d6_locked25](competition/experiments/mlp_event_cat_sgns8e640_context24_d6_locked25/result.json) |
| v2_wide256_s2026_locked25 | 2025-01-25 | 0.816084 | 0.193254 | 29.2 | [v2_wide256_s2026_locked25](competition/experiments/v2_wide256_s2026_locked25/result.json) |
| mlp_event_cat_sgns8_s2026_context24_d6_locked25 | 2025-01-25 | 0.815970 | 0.193477 | 42.2 | [mlp_event_cat_sgns8_s2026_context24_d6_locked25](competition/experiments/mlp_event_cat_sgns8_s2026_context24_d6_locked25/result.json) |
| v2_cross256_s2026_locked25 | 2025-01-25 | 0.815776 | 0.193348 | 20.5 | [v2_cross256_s2026_locked25](competition/experiments/v2_cross256_s2026_locked25/result.json) |
| event_cat_sgns8e640_context24_d6_locked25 | 2025-01-25 | 0.815512 | 0.193082 | 12.6 | [event_cat_sgns8e640_context24_d6_locked25](competition/experiments/event_cat_sgns8e640_context24_d6_locked25/result.json) |
| cat_sgns8e640_context24_d6_locked25 | 2025-01-25 | 0.813338 | 0.194205 | 706.3 | [cat_sgns8e640_context24_d6_locked25](competition/experiments/cat_sgns8e640_context24_d6_locked25/result.json) |
| mlp_context_cat_sgns8e320_context24_d6_locked25 | 2025-01-25 | 0.812771 | 0.194204 | 32.2 | [mlp_context_cat_sgns8e320_context24_d6_locked25](competition/experiments/mlp_context_cat_sgns8e320_context24_d6_locked25/result.json) |
| cat_sgns8_s2026_context24_d6_locked25 | 2025-01-25 | 0.812736 | 0.194315 | 617.1 | [cat_sgns8_s2026_context24_d6_locked25](competition/experiments/cat_sgns8_s2026_context24_d6_locked25/result.json) |
| mlp_event_cat_sgns8e320_context24_d6_locked25 | 2025-01-25 | 0.812416 | 0.194500 | 23.5 | [mlp_event_cat_sgns8e320_context24_d6_locked25](competition/experiments/mlp_event_cat_sgns8e320_context24_d6_locked25/result.json) |
| mlp_context_cat_sgns8long_context_24_180000_d6_locked25 | 2025-01-25 | 0.808980 | 0.195409 | 34.0 | [mlp_context_cat_sgns8long_context_24_180000_d6_locked25](competition/experiments/mlp_context_cat_sgns8long_context_24_180000_d6_locked25/result.json) |
| cat_sgns8e320_context24_d6_locked25 | 2025-01-25 | 0.808800 | 0.195470 | 620.3 | [cat_sgns8e320_context24_d6_locked25](competition/experiments/cat_sgns8e320_context24_d6_locked25/result.json) |
| mlp_event_long_ema_s2026_d0.1_locked25 | 2025-01-25 | 0.808726 | 0.195288 | 31.4 | [mlp_event_long_ema_s2026_d0.1_locked25](competition/experiments/mlp_event_long_ema_s2026_d0.1_locked25/result.json) |
| mlp_event_ce_cat_sgns8long_context_24_180000_d6_locked25 | 2025-01-25 | 0.807945 | 0.196134 | 23.6 | [mlp_event_ce_cat_sgns8long_context_24_180000_d6_locked25](competition/experiments/mlp_event_ce_cat_sgns8long_context_24_180000_d6_locked25/result.json) |
| feature_reference_parity_smoke25 | 2025-01-25 | 0.807735 | 0.197896 | 11.9 | [feature_reference_parity_smoke25](competition/experiments/feature_reference_parity_smoke25/result.json) |
| cat_sgns8long_context_24_180000_d6_locked25 | 2025-01-25 | 0.804098 | 0.196867 | 487.7 | [cat_sgns8long_context_24_180000_d6_locked25](competition/experiments/cat_sgns8long_context_24_180000_d6_locked25/result.json) |
| event_cat_sgns8_16_context_24_180000_d6_locked25 | 2025-01-25 | 0.802714 | 0.196850 | 14.4 | [event_cat_sgns8_16_context_24_180000_d6_locked25](competition/experiments/event_cat_sgns8_16_context_24_180000_d6_locked25/result.json) |
| cat_sgns8_16_context_24_180000_d6_locked25 | 2025-01-25 | 0.800494 | 0.197801 | 847.6 | [cat_sgns8_16_context_24_180000_d6_locked25](competition/experiments/cat_sgns8_16_context_24_180000_d6_locked25/result.json) |
| sgns8_locked_cut25 | 2025-01-25 | 0.763408 | 0.220270 | 327.4 | [sgns8_locked_cut25](competition/experiments/sgns8_locked_cut25/result.json) |
| v3_wide512_s73 | 2025-01-26 | 0.829267 | 0.188581 | 73.2 | [v3_wide512_s73](competition/experiments/v3_wide512_s73/result.json) |
| v2_multiview_wide256_s2026 | 2025-01-26 | 0.829246 | 0.188614 | 49.2 | [v2_multiview_wide256_s2026](competition/experiments/v2_multiview_wide256_s2026/result.json) |
| v3_cross512_s2027 | 2025-01-26 | 0.829132 | 0.188463 | 55.7 | [v3_cross512_s2027](competition/experiments/v3_cross512_s2027/result.json) |
| v2_multiview_cross256_s2026 | 2025-01-26 | 0.829103 | 0.188630 | 44.4 | [v2_multiview_cross256_s2026](competition/experiments/v2_multiview_cross256_s2026/result.json) |
| mlp_event_cat_sgns8e640_context24_d6 | 2025-01-26 | 0.828389 | 0.189046 | 46.4 | [mlp_event_cat_sgns8e640_context24_d6](competition/experiments/mlp_event_cat_sgns8e640_context24_d6/result.json) |
| v2_wide256_s2026 | 2025-01-26 | 0.827959 | 0.189117 | 50.4 | [v2_wide256_s2026](competition/experiments/v2_wide256_s2026/result.json) |
| mlp_event_cat_sgns8e320_context24_d6 | 2025-01-26 | 0.827889 | 0.189212 | 56.1 | [mlp_event_cat_sgns8e320_context24_d6](competition/experiments/mlp_event_cat_sgns8e320_context24_d6/result.json) |
| mlp_event_cat_sgns8_s2026_context24_d6 | 2025-01-26 | 0.827855 | 0.189206 | 57.2 | [mlp_event_cat_sgns8_s2026_context24_d6](competition/experiments/mlp_event_cat_sgns8_s2026_context24_d6/result.json) |
| v2_cross256_s2026 | 2025-01-26 | 0.827848 | 0.189081 | 43.9 | [v2_cross256_s2026](competition/experiments/v2_cross256_s2026/result.json) |
| mlp_context_cat_sgns8e320_context24_d6 | 2025-01-26 | 0.827841 | 0.189121 | 48.4 | [mlp_context_cat_sgns8e320_context24_d6](competition/experiments/mlp_context_cat_sgns8e320_context24_d6/result.json) |
| v2_attention128_s2026 | 2025-01-26 | 0.827647 | 0.189211 | 164.2 | [v2_attention128_s2026](competition/experiments/v2_attention128_s2026/result.json) |
| event_cat_sgns8e640_context24_d6 | 2025-01-26 | 0.827592 | 0.189034 | 19.0 | [event_cat_sgns8e640_context24_d6](competition/experiments/event_cat_sgns8e640_context24_d6/result.json) |
| event_cat_sgns8e320_context24_d6 | 2025-01-26 | 0.827157 | 0.189190 | 26.0 | [event_cat_sgns8e320_context24_d6](competition/experiments/event_cat_sgns8e320_context24_d6/result.json) |
| event_cat_sgns8_s2026_context24_d6 | 2025-01-26 | 0.826707 | 0.189285 | 22.8 | [event_cat_sgns8_s2026_context24_d6](competition/experiments/event_cat_sgns8_s2026_context24_d6/result.json) |
| mlp_aligned_s2026_e320 | 2025-01-26 | 0.826631 | 0.189440 | 40.5 | [mlp_aligned_s2026_e320](competition/experiments/mlp_aligned_s2026_e320/result.json) |
| cat_sgns8e640_context24_d6 | 2025-01-26 | 0.825058 | 0.190112 | 559.4 | [cat_sgns8e640_context24_d6](competition/experiments/cat_sgns8e640_context24_d6/result.json) |
| cat_sgns8e320_context24_d6 | 2025-01-26 | 0.824595 | 0.190208 | 441.0 | [cat_sgns8e320_context24_d6](competition/experiments/cat_sgns8e320_context24_d6/result.json) |
| cat_sgns8_s2026_context24_d6 | 2025-01-26 | 0.824114 | 0.190323 | 366.4 | [cat_sgns8_s2026_context24_d6](competition/experiments/cat_sgns8_s2026_context24_d6/result.json) |
| mlp_context_cat_sgns8long_context_24_180000_d6 | 2025-01-26 | 0.823675 | 0.190453 | 55.7 | [mlp_context_cat_sgns8long_context_24_180000_d6](competition/experiments/mlp_context_cat_sgns8long_context_24_180000_d6/result.json) |
| mlp_event_long_ema_s2026_d0.1 | 2025-01-26 | 0.823496 | 0.190564 | 75.1 | [mlp_event_long_ema_s2026_d0.1](competition/experiments/mlp_event_long_ema_s2026_d0.1/result.json) |
| mlp_event_cat_foldin_long24_d6 | 2025-01-26 | 0.823455 | 0.190645 | 47.1 | [mlp_event_cat_foldin_long24_d6](competition/experiments/mlp_event_cat_foldin_long24_d6/result.json) |
| mlp_event_long_ema_s7_d0.2 | 2025-01-26 | 0.823352 | 0.190696 | 61.5 | [mlp_event_long_ema_s7_d0.2](competition/experiments/mlp_event_long_ema_s7_d0.2/result.json) |
| mlp_event_ce_cat_sgns8long_context_24_180000_d6 | 2025-01-26 | 0.823270 | 0.190653 | 52.4 | [mlp_event_ce_cat_sgns8long_context_24_180000_d6](competition/experiments/mlp_event_ce_cat_sgns8long_context_24_180000_d6/result.json) |
| mlp_event_long_constzero | 2025-01-26 | 0.823270 | 0.190653 | 45.9 | [mlp_event_long_constzero](competition/experiments/mlp_event_long_constzero/result.json) |
| mlp_event_long_ema_s42_d0.2 | 2025-01-26 | 0.823224 | 0.190686 | 43.9 | [mlp_event_long_ema_s42_d0.2](competition/experiments/mlp_event_long_ema_s42_d0.2/result.json) |
| event_cat_sgns8long_context_24_180000_d6 | 2025-01-26 | 0.822218 | 0.190720 | 20.4 | [event_cat_sgns8long_context_24_180000_d6](competition/experiments/event_cat_sgns8long_context_24_180000_d6/result.json) |
| event_cat_sgns8long_context_24_180000_d8 | 2025-01-26 | 0.822134 | 0.190752 | 18.4 | [event_cat_sgns8long_context_24_180000_d8](competition/experiments/event_cat_sgns8long_context_24_180000_d8/result.json) |
| mlp_bce_cat_sgns8long_context_24_180000_d6 | 2025-01-26 | 0.820638 | 0.191969 | 108.4 | [mlp_bce_cat_sgns8long_context_24_180000_d6](competition/experiments/mlp_bce_cat_sgns8long_context_24_180000_d6/result.json) |
| cat_foldin_long24_d6 | 2025-01-26 | 0.819896 | 0.191627 | 292.9 | [cat_foldin_long24_d6](competition/experiments/cat_foldin_long24_d6/result.json) |
| cat_sgns8long_context_24_180000_d6 | 2025-01-26 | 0.819746 | 0.191664 | 431.4 | [cat_sgns8long_context_24_180000_d6](competition/experiments/cat_sgns8long_context_24_180000_d6/result.json) |
| cat_sgns8long_context_24_180000_d8 | 2025-01-26 | 0.819593 | 0.191675 | 235.3 | [cat_sgns8long_context_24_180000_d8](competition/experiments/cat_sgns8long_context_24_180000_d8/result.json) |
| mlp_event_ce_cat_sgns8_16_context_24_180000_d6 | 2025-01-26 | 0.815565 | 0.193021 | 71.5 | [mlp_event_ce_cat_sgns8_16_context_24_180000_d6](competition/experiments/mlp_event_ce_cat_sgns8_16_context_24_180000_d6/result.json) |
| event_cat_rolling24_sgns8_16_d8 | 2025-01-26 | 0.815408 | 0.192757 | 34.6 | [event_cat_rolling24_sgns8_16_d8](competition/experiments/event_cat_rolling24_sgns8_16_d8/result.json) |
| event_cat_rolling24_sgns8_16_d6 | 2025-01-26 | 0.815174 | 0.192790 | 32.2 | [event_cat_rolling24_sgns8_16_d6](competition/experiments/event_cat_rolling24_sgns8_16_d6/result.json) |
| event_cat_sgns8_16_context_24_180000_d6 | 2025-01-26 | 0.814354 | 0.193045 | 22.2 | [event_cat_sgns8_16_context_24_180000_d6](competition/experiments/event_cat_sgns8_16_context_24_180000_d6/result.json) |
| event_cat_sgns8_16_context_24_180000_d8 | 2025-01-26 | 0.814242 | 0.193130 | 25.5 | [event_cat_sgns8_16_context_24_180000_d8](competition/experiments/event_cat_sgns8_16_context_24_180000_d8/result.json) |
| mlp_bce_cat_sgns8_16_context_24_180000_d6 | 2025-01-26 | 0.812872 | 0.194183 | 109.9 | [mlp_bce_cat_sgns8_16_context_24_180000_d6](competition/experiments/mlp_bce_cat_sgns8_16_context_24_180000_d6/result.json) |
| cat_rolling24_sgns8_16_d8 | 2025-01-26 | 0.812830 | 0.193645 | 383.4 | [cat_rolling24_sgns8_16_d8](competition/experiments/cat_rolling24_sgns8_16_d8/result.json) |
| cat_rolling24_sgns8_16_d6 | 2025-01-26 | 0.812605 | 0.193630 | 365.2 | [cat_rolling24_sgns8_16_d6](competition/experiments/cat_rolling24_sgns8_16_d6/result.json) |
| cat_sgns8_16_context_24_180000_d8 | 2025-01-26 | 0.811700 | 0.194019 | 524.9 | [cat_sgns8_16_context_24_180000_d8](competition/experiments/cat_sgns8_16_context_24_180000_d8/result.json) |
| cat_sgns8_16_context_24_180000_d6 | 2025-01-26 | 0.811689 | 0.193953 | 350.0 | [cat_sgns8_16_context_24_180000_d6](competition/experiments/cat_sgns8_16_context_24_180000_d6/result.json) |
| event_cat_sgns8_16_context_22_80000_d6 | 2025-01-26 | 0.804101 | 0.196112 | 12.1 | [event_cat_sgns8_16_context_22_80000_d6](competition/experiments/event_cat_sgns8_16_context_22_80000_d6/result.json) |
| sgns8_const_n8_binary_e640_s42 | 2025-01-26 | 0.803856 | 0.216255 | 704.1 | [sgns8_const_n8_binary_e640_s42](competition/experiments/sgns8_const_n8_binary_e640_s42/result.json) |
| cat_sgns8e80_16_context_22_80000_d6 | 2025-01-26 | 0.802837 | 0.196706 | 281.1 | [cat_sgns8e80_16_context_22_80000_d6](competition/experiments/cat_sgns8e80_16_context_22_80000_d6/result.json) |
| sgns8_const_n8_binary_e320_s42 | 2025-01-26 | 0.802547 | 0.216460 | 614.6 | [sgns8_const_n8_binary_e320_s42](competition/experiments/sgns8_const_n8_binary_e320_s42/result.json) |
| warm_ctr_anchor01 | 2025-01-26 | 0.802447 | 0.221422 | 60.3 | [warm_ctr_anchor01](competition/experiments/warm_ctr_anchor01/result.json) |
| sgns8_const_e320_s2026 | 2025-01-26 | 0.802377 | 0.216488 | 311.4 | [sgns8_const_e320_s2026](competition/experiments/sgns8_const_e320_s2026/result.json) |
| cat_sgns8_16_context_22_80000_d6 | 2025-01-26 | 0.801327 | 0.197110 | 224.2 | [cat_sgns8_16_context_22_80000_d6](competition/experiments/cat_sgns8_16_context_22_80000_d6/result.json) |
| sgns8_const_e160_s42 | 2025-01-26 | 0.795586 | 0.217186 | 549.5 | [sgns8_const_e160_s42](competition/experiments/sgns8_const_e160_s42/result.json) |
| cat_sgns_context_24_180000_d8 | 2025-01-26 | 0.794522 | 0.198822 | 368.6 | [cat_sgns_context_24_180000_d8](competition/experiments/cat_sgns_context_24_180000_d8/result.json) |
| cat_sgns_context_24_180000_d6 | 2025-01-26 | 0.794402 | 0.198748 | 352.8 | [cat_sgns_context_24_180000_d6](competition/experiments/cat_sgns_context_24_180000_d6/result.json) |
| cat_sgns_context_22_80000_d6 | 2025-01-26 | 0.789505 | 0.200495 | 210.4 | [cat_sgns_context_22_80000_d6](competition/experiments/cat_sgns_context_22_80000_d6/result.json) |
| sgns8_const_n8_distinct_clicks_e160_s42 | 2025-01-26 | 0.787164 | 0.217815 | 264.0 | [sgns8_const_n8_distinct_clicks_e160_s42](competition/experiments/sgns8_const_n8_distinct_clicks_e160_s42/result.json) |
| sgns8_distinct_e80_s42 | 2025-01-26 | 0.785209 | 0.217915 | 265.5 | [sgns8_distinct_e80_s42](competition/experiments/sgns8_distinct_e80_s42/result.json) |
| sgns8_n8_e80_s42 | 2025-01-26 | 0.783731 | 0.218100 | 212.6 | [sgns8_n8_e80_s42](competition/experiments/sgns8_n8_e80_s42/result.json) |
| sgns8_const_n4_binary_e160_s42 | 2025-01-26 | 0.780315 | 0.218959 | 197.7 | [sgns8_const_n4_binary_e160_s42](competition/experiments/sgns8_const_n4_binary_e160_s42/result.json) |
| sgns12_n8_e80_s42 | 2025-01-26 | 0.777752 | 0.218331 | 118.2 | [sgns12_n8_e80_s42](competition/experiments/sgns12_n8_e80_s42/result.json) |
| sgns8_n8_e80_s2026 | 2025-01-26 | 0.776602 | 0.218768 | 258.8 | [sgns8_n8_e80_s2026](competition/experiments/sgns8_n8_e80_s2026/result.json) |
| cat_context_24_180000_d6 | 2025-01-26 | 0.775427 | 0.203946 | 234.9 | [cat_context_24_180000_d6](competition/experiments/cat_context_24_180000_d6/result.json) |
| sgns8_n16_e80_s42 | 2025-01-26 | 0.774388 | 0.218823 | 263.3 | [sgns8_n16_e80_s42](competition/experiments/sgns8_n16_e80_s42/result.json) |
| sgns12_const_n8_binary_e160_s42 | 2025-01-26 | 0.773477 | 0.219972 | 119.6 | [sgns12_const_n8_binary_e160_s42](competition/experiments/sgns12_const_n8_binary_e160_s42/result.json) |
| cat_als16_32_200000_d8 | 2025-01-26 | 0.772891 | 0.206095 | 176.1 | [cat_als16_32_200000_d8](competition/experiments/cat_als16_32_200000_d8/result.json) |
| cat_als16_32_200000_d6 | 2025-01-26 | 0.772555 | 0.205889 | 184.0 | [cat_als16_32_200000_d6](competition/experiments/cat_als16_32_200000_d6/result.json) |
| cat_als16_32_80000_d6 | 2025-01-26 | 0.772018 | 0.205678 | 101.9 | [cat_als16_32_80000_d6](competition/experiments/cat_als16_32_80000_d6/result.json) |
| sgns16_n16_e80_s42 | 2025-01-26 | 0.756626 | 0.222494 | 159.9 | [sgns16_n16_e80_s42](competition/experiments/sgns16_n16_e80_s42/result.json) |
| sgns6_const_n8_binary_e160_s42 | 2025-01-26 | 0.755133 | 0.233233 | 189.0 | [sgns6_const_n8_binary_e160_s42](competition/experiments/sgns6_const_n8_binary_e160_s42/result.json) |
| sgns16_n8_e80_s42 | 2025-01-26 | 0.752524 | 0.224202 | 158.0 | [sgns16_n8_e80_s42](competition/experiments/sgns16_n8_e80_s42/result.json) |
| sgns6_n8_e80_s42 | 2025-01-26 | 0.744271 | 0.233998 | 122.7 | [sgns6_n8_e80_s42](competition/experiments/sgns6_n8_e80_s42/result.json) |
| sgns16_mps | 2025-01-26 | 0.743287 | 0.222836 | 45.1 | [sgns16_mps](competition/experiments/sgns16_mps/result.json) |
| sgns32_n8_e80_s42 | 2025-01-26 | 0.710757 | 0.236623 | 229.7 | [sgns32_n8_e80_s42](competition/experiments/sgns32_n8_e80_s42/result.json) |
| sgns4_n8_e80_s42 | 2025-01-26 | 0.697389 | 0.238997 | 118.0 | [sgns4_n8_e80_s42](competition/experiments/sgns4_n8_e80_s42/result.json) |
| als16_a10_r0.1 | 2025-01-26 | 0.677857 | 0.612107 | 6.6 | [als16_a10_r0.1](competition/experiments/als16_a10_r0.1/result.json) |
| als32_a10_r0.1 | 2025-01-26 | 0.672467 | 0.720203 | 9.5 | [als32_a10_r0.1](competition/experiments/als32_a10_r0.1/result.json) |
| als64_a20 | 2025-01-26 | 0.671045 | 1.431431 | 18.4 | [als64_a20](competition/experiments/als64_a20/result.json) |
| als24_a5_r0.1 | 2025-01-26 | 0.657132 | 0.438380 | 8.0 | [als24_a5_r0.1](competition/experiments/als24_a5_r0.1/result.json) |
| als128_a40_r0.5 | 2025-01-26 | 0.655075 | 0.669661 | 36.8 | [als128_a40_r0.5](competition/experiments/als128_a40_r0.5/result.json) |
| als64_a5_r0.1 | 2025-01-26 | 0.654374 | 0.515312 | 16.5 | [als64_a5_r0.1](competition/experiments/als64_a5_r0.1/result.json) |
| als128_a10_r0.1 | 2025-01-26 | 0.651288 | 0.900175 | 36.4 | [als128_a10_r0.1](competition/experiments/als128_a10_r0.1/result.json) |
| als128_a5_r0.1 | 2025-01-26 | 0.644371 | 0.525476 | 36.6 | [als128_a5_r0.1](competition/experiments/als128_a5_r0.1/result.json) |
| als256_a20_r0.5 | 2025-01-26 | 0.640905 | 0.479180 | 105.5 | [als256_a20_r0.5](competition/experiments/als256_a20_r0.5/result.json) |
| cat200000_d6_l20 | 2025-01-26 | 0.636817 | 0.225038 | 49.2 | [cat200000_d6_l20](competition/experiments/cat200000_d6_l20/result.json) |
| cat200000_d8_l30 | 2025-01-26 | 0.636023 | 0.224903 | 30.8 | [cat200000_d8_l30](competition/experiments/cat200000_d8_l30/result.json) |
| cat80000_d4_l20 | 2025-01-26 | 0.635777 | 0.224988 | 19.9 | [cat80000_d4_l20](competition/experiments/cat80000_d4_l20/result.json) |
| cat80k_d6 | 2025-01-26 | 0.635350 | 0.225122 | 25.6 | [cat80k_d6](competition/experiments/cat80k_d6/result.json) |
| als16_a3_r0.1 | 2025-01-26 | 0.634656 | 0.314382 | 6.6 | [als16_a3_r0.1](competition/experiments/als16_a3_r0.1/result.json) |
| als8_a10_r0.1 | 2025-01-26 | 0.630693 | 0.575595 | 5.7 | [als8_a10_r0.1](competition/experiments/als8_a10_r0.1/result.json) |
| cf_26_fp64_audit | 2025-01-26 | 0.615704 | 0.874752 | 1.6 | [cf_26_fp64_audit](competition/experiments/cf_26_fp64_audit/result.json) |
| cf_26 | 2025-01-26 | 0.615649 | 0.542159 | 14.6 | [cf_26](competition/experiments/cf_26/result.json) |

## 最终交付

- 推荐提交：[competition/final_v2/submission.csv](competition/final_v2/submission.csv)
- 模型与复现配置：[competition/final_v2/versions/20261024_110038](competition/final_v2/versions/20261024_110038)

```json
{
  "submission": "competition/final_v2/submission.csv",
  "model_dir": "competition/final_v2/versions/20261024_110038",
  "sources": [
    "event_cat_sgns8e640_context24_d6",
    "mlp_event_cat_sgns8_s2026_context24_d6",
    "v2_multiview_wide256_s2026",
    "v2_multiview_cross256_s2026"
  ],
  "weights": [
    0.25,
    0.25,
    0.25,
    0.25
  ],
  "blend_kind": "logit",
  "validation_26": {
    "auc": 0.8293272082809982,
    "logloss": 0.18853976902343578,
    "average_precision": 0.28889980792100434,
    "accuracy_at_0_5": 0.9376088565086231,
    "rows": 351380
  },
  "validation_25": {
    "auc": 0.8192463525707806,
    "logloss": 0.1922149621174674,
    "average_precision": 0.2708208642983001,
    "accuracy_at_0_5": 0.9372281866982076,
    "rows": 439640
  },
  "validation_24": {
    "auc": 0.8054023107836017,
    "logloss": 0.19540901529588606,
    "average_precision": 0.2529105015069471,
    "accuracy_at_0_5": 0.937416541478885,
    "rows": 479280
  },
  "audit_24_sources": [
    "event_cat_sgns8e640_context24_d6_locked24",
    "mlp_event_cat_sgns8_s2026_context24_d6_locked24",
    "v2_multiview_wide256_s2026_locked24",
    "v2_multiview_cross256_s2026_locked24"
  ],
  "validation_protocol": "Jan26 used for hyperparameters; Jan26+Jan25 cohort-weighted AUC selects coarse blend; Jan24 recipe audit is not used for selection",
  "rows": 529390,
  "sha256": "677392c95cf46b7ec2640e73d0bf6ed7868f4550562a2ff0367f6e330d7bee80",
  "csv_checks_passed": true,
  "official_evaluation_labels_available": false,
  "selection_score": 0.8186225690006399,
  "cache_dependencies": "data/cache_v1 holds ID mappings and cached event arrays; learned inference representations are bundled in the version directory; competition/cache holds regenerable feature matrices",
  "representations": "competition/final_v2/versions/20261024_110038/representations",
  "reproduce": "python3 arena_finalize.py --final-dir competition/final_v2 --predict-only"
}
```

## 检索增强的自迭代

检索对象是论文与官方文档摘要、数据诊断和本地实验记录。配置生成器检索相近问题，结合已有指标选择下一组参数；检索内容与理由保存到每个实验目录。推荐模型自身负责数值点击预测，避免将不必要的语言模型推理加入每条样本。

研究资料保存在 `competition/research/`；参考原始论文与官方文档，实验优劣以本地时间验证为准。配置生成使用本地 TF-IDF 检索与基于指标的规则；未调用付费语言模型 API。

主要原始资料：[时间泄漏研究](https://arxiv.org/abs/2010.11060)、[implicit ALS 文档](https://benfred.github.io/implicit/api/models/cpu/als.html)、[Item2Vec](https://arxiv.org/abs/1603.04259)、[DIN](https://arxiv.org/abs/1706.06978)、[LightGCN](https://arxiv.org/abs/2002.02126)。具体借鉴及未采用原因见每轮 `retrieval_plan.json` 与执行记录。

## 全部执行与取舍记录

- 2026-10-24T01:49:01+08:00 [progress] 启动自主优化：复核比赛要求；本机 M2/8核/16GB，MPS 可用，电源接通；旧完整双塔 AUC 0.585356。保留所有既有产物。
- 2026-10-24T01:49:01+08:00 [research] 检索原始论文 EASE、Google 双塔负采样及 implicit/CatBoost 官方文档；先验证官方共点击方案，再根据结果分配算力。
- 2026-10-24T01:49:13+08:00 [progress] 构建 2025-01-26 严格前向快照：588,873 个事件、841,074 个去重用户物品交互、21,111,353 个共点击非零项，耗时 8.7 秒。
- 2026-10-24T01:49:19+08:00 [experiment] 实验 cf_26：AUC 0.615649，官方三信号公式 AUC 0.615649，最佳 starter_cf_w25。
- 2026-10-24T01:52:20+08:00 [progress] 构建 2025-01-22 严格前向快照：197,017 个事件、587,527 个去重用户物品交互、10,118,807 个共点击非零项，耗时 4.2 秒。
- 2026-10-24T01:52:34+08:00 [experiment] 实验 als64_a20：implicit_als，切点 2025-01-26，AUC 0.671045，Logloss 1.431431，耗时 18.4 秒。
- 2026-10-24T01:52:42+08:00 [experiment] 实验 cat80k_d6：catboost_numeric，切点 2025-01-26，AUC 0.635350，Logloss 0.225122，耗时 25.6 秒。
- 2026-10-24T01:55:54+08:00 [scheduler] Go 调度启动 PID=12093，最大新实验 9，预算 2.00 小时。
- 2026-10-24T01:55:56+08:00 [retrieval] als32_a10_r0.1: Current best ALS AUC=0.671045; numeric tree AUC=0.635350. Test als capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T01:56:07+08:00 [experiment] 实验 als32_a10_r0.1：implicit_als，切点 2025-01-26，AUC 0.672467，Logloss 0.720203，耗时 9.5 秒。
- 2026-10-24T01:56:07+08:00 [scheduler] als32_a10_r0.1 完成，耗时 11.6 秒。
- 2026-10-24T01:56:08+08:00 [retrieval] als128_a10_r0.1: Current best ALS AUC=0.672467; numeric tree AUC=0.635350. Test als capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T01:56:47+08:00 [experiment] 实验 als128_a10_r0.1：implicit_als，切点 2025-01-26，AUC 0.651288，Logloss 0.900175，耗时 36.4 秒。
- 2026-10-24T01:56:47+08:00 [scheduler] als128_a10_r0.1 完成，耗时 38.5 秒。
- 2026-10-24T01:56:48+08:00 [retrieval] als128_a40_r0.5: Current best ALS AUC=0.672467; numeric tree AUC=0.635350. Test als capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T01:57:27+08:00 [experiment] 实验 als128_a40_r0.5：implicit_als，切点 2025-01-26，AUC 0.655075，Logloss 0.669661，耗时 36.8 秒。
- 2026-10-24T01:57:27+08:00 [scheduler] als128_a40_r0.5 完成，耗时 38.9 秒。
- 2026-10-24T01:57:28+08:00 [retrieval] als256_a20_r0.5: Current best ALS AUC=0.672467; numeric tree AUC=0.635350. Test als capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T01:59:15+08:00 [experiment] 实验 als256_a20_r0.5：implicit_als，切点 2025-01-26，AUC 0.640905，Logloss 0.479180，耗时 105.5 秒。
- 2026-10-24T01:59:16+08:00 [scheduler] als256_a20_r0.5 完成，耗时 107.6 秒。
- 2026-10-24T01:59:17+08:00 [retrieval] als64_a5_r0.1: Current best ALS AUC=0.672467; numeric tree AUC=0.635350. Test als capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T01:59:35+08:00 [experiment] 实验 als64_a5_r0.1：implicit_als，切点 2025-01-26，AUC 0.654374，Logloss 0.515312，耗时 16.5 秒。
- 2026-10-24T01:59:35+08:00 [scheduler] als64_a5_r0.1 完成，耗时 18.6 秒。
- 2026-10-24T01:59:36+08:00 [retrieval] als128_a5_r0.1: Current best ALS AUC=0.672467; numeric tree AUC=0.635350. Test als capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:00:15+08:00 [experiment] 实验 als128_a5_r0.1：implicit_als，切点 2025-01-26，AUC 0.644371，Logloss 0.525476，耗时 36.6 秒。
- 2026-10-24T02:00:15+08:00 [scheduler] als128_a5_r0.1 完成，耗时 38.5 秒。
- 2026-10-24T02:00:16+08:00 [retrieval] cat80000_d4_l20: Current best ALS AUC=0.672467; numeric tree AUC=0.635350. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:00:25+08:00 [research] Go 调度已实际运行，按 TF-IDF 检索论文与本地实验记录后生成参数配置。低秩 ALS 优于高秩的初步证据触发新增 8/16/24 维实验；准备把标准化潜在匹配分数加入严格前向训练的 CatBoost。
- 2026-10-24T02:00:38+08:00 [experiment] 实验 cat80000_d4_l20：catboost_numeric，切点 2025-01-26，AUC 0.635777，Logloss 0.224988，耗时 19.9 秒。
- 2026-10-24T02:00:38+08:00 [scheduler] cat80000_d4_l20 完成，耗时 22.2 秒。
- 2026-10-24T02:00:39+08:00 [retrieval] als16_a10_r0.1: Current best ALS AUC=0.672467; numeric tree AUC=0.635777. Test als capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:00:47+08:00 [experiment] 实验 als16_a10_r0.1：implicit_als，切点 2025-01-26，AUC 0.677857，Logloss 0.612107，耗时 6.6 秒。
- 2026-10-24T02:00:48+08:00 [scheduler] als16_a10_r0.1 完成，耗时 8.6 秒。
- 2026-10-24T02:00:49+08:00 [retrieval] als16_a3_r0.1: Current best ALS AUC=0.677857; numeric tree AUC=0.635777. Test als capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:00:57+08:00 [experiment] 实验 als16_a3_r0.1：implicit_als，切点 2025-01-26，AUC 0.634656，Logloss 0.314382，耗时 6.6 秒。
- 2026-10-24T02:00:57+08:00 [scheduler] als16_a3_r0.1 完成，耗时 8.6 秒。
- 2026-10-24T02:00:57+08:00 [scheduler] 本轮 Go 调度结束；所有日志、配置、检索依据和指标已保存。
- 2026-10-24T02:02:46+08:00 [research] 依据检索到的辅助隐式反馈与负采样研究，启动 MPS GPU 的低秩负采样表示学习，和 CPU ALS 形成不同归纳偏置，验证是否值得融合。
- 2026-10-24T02:03:33+08:00 [experiment] GPU 辅助反馈实验 sgns16_mps：设备 mps，AUC 0.743287，最佳 epoch 20，耗时 45.1 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:03:46+08:00 [scheduler] Go 调度启动 PID=16803，最大新实验 30，预算 3.00 小时。
- 2026-10-24T02:03:47+08:00 [retrieval] cat_als16_32_80000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.635777. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:05:31+08:00 [experiment] 实验 cat_als16_32_80000_d6：catboost_numeric，切点 2025-01-26，AUC 0.772018，Logloss 0.205678，耗时 101.9 秒。
- 2026-10-24T02:05:32+08:00 [scheduler] cat_als16_32_80000_d6 完成，耗时 104.3 秒。
- 2026-10-24T02:05:33+08:00 [retrieval] als8_a10_r0.1: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test als capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:05:40+08:00 [experiment] 实验 als8_a10_r0.1：implicit_als，切点 2025-01-26，AUC 0.630693，Logloss 0.575595，耗时 5.7 秒。
- 2026-10-24T02:05:41+08:00 [scheduler] als8_a10_r0.1 完成，耗时 7.9 秒。
- 2026-10-24T02:05:42+08:00 [retrieval] als24_a5_r0.1: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test als capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:05:51+08:00 [experiment] 实验 als24_a5_r0.1：implicit_als，切点 2025-01-26，AUC 0.657132，Logloss 0.438380，耗时 8.0 秒。
- 2026-10-24T02:05:52+08:00 [scheduler] als24_a5_r0.1 完成，耗时 9.9 秒。
- 2026-10-24T02:05:53+08:00 [retrieval] cat200000_d6_l20: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:06:44+08:00 [experiment] 实验 cat200000_d6_l20：catboost_numeric，切点 2025-01-26，AUC 0.636817，Logloss 0.225038，耗时 49.2 秒。
- 2026-10-24T02:06:44+08:00 [scheduler] cat200000_d6_l20 完成，耗时 51.6 秒。
- 2026-10-24T02:06:45+08:00 [retrieval] cat200000_d8_l30: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:07:18+08:00 [experiment] 实验 cat200000_d8_l30：catboost_numeric，切点 2025-01-26，AUC 0.636023，Logloss 0.224903，耗时 30.8 秒。
- 2026-10-24T02:07:19+08:00 [scheduler] cat200000_d8_l30 完成，耗时 33.3 秒。
- 2026-10-24T02:07:20+08:00 [retrieval] cat_als16_32_200000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:07:59+08:00 [scheduler] Go 调度启动 PID=19466，通道 gpu，每通道最大新实验 8，预算 2.00 小时。
- 2026-10-24T02:08:01+08:00 [retrieval] sgns16_n8_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:08:04+08:00 [failure] sgns16_n8_e80_s42: exit status 1
- 2026-10-24T02:08:05+08:00 [retrieval] sgns32_n8_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:08:08+08:00 [failure] sgns32_n8_e80_s42: exit status 1
- 2026-10-24T02:08:09+08:00 [retrieval] sgns8_n8_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:08:12+08:00 [failure] sgns8_n8_e80_s42: exit status 1
- 2026-10-24T02:08:14+08:00 [retrieval] sgns16_n16_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:08:16+08:00 [failure] sgns16_n16_e80_s42: exit status 1
- 2026-10-24T02:08:18+08:00 [retrieval] sgns16_n8_e80_s2026: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:08:21+08:00 [failure] sgns16_n8_e80_s2026: exit status 1
- 2026-10-24T02:08:22+08:00 [retrieval] sgns32_n8_e80_s2026: Current best ALS AUC=0.677857; numeric tree AUC=0.772018. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:08:25+08:00 [failure] sgns32_n8_e80_s2026: exit status 1
- 2026-10-24T02:08:27+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T02:08:27+08:00 [scheduler] 本轮 Go 调度结束；所有日志、配置、检索依据和指标已保存。
- 2026-10-24T02:10:26+08:00 [experiment] 实验 cat_als16_32_200000_d6：catboost_numeric，切点 2025-01-26，AUC 0.772555，Logloss 0.205889，耗时 184.0 秒。
- 2026-10-24T02:10:26+08:00 [scheduler] cat_als16_32_200000_d6 完成，耗时 186.4 秒。
- 2026-10-24T02:10:27+08:00 [retrieval] cat_als16_32_200000_d8: Current best ALS AUC=0.677857; numeric tree AUC=0.772555. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:11:59+08:00 [fix] GPU 批量调度首次失败，原因是设置高水位 0.55 时未同步降低 PyTorch 默认低水位 1.4。已保留六次启动失败日志，修正低水位为 0.4，新增通道启动时的实际 MPS 分配健康检查，再重试。模型训练本身未发生 OOM。
- 2026-10-24T02:12:00+08:00 [scheduler] Go 调度启动 PID=22508，通道 gpu，每通道最大新实验 8，预算 2.00 小时。
- 2026-10-24T02:12:03+08:00 [retrieval] sgns16_n8_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.772555. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:13:25+08:00 [experiment] 实验 cat_als16_32_200000_d8：catboost_numeric，切点 2025-01-26，AUC 0.772891，Logloss 0.206095，耗时 176.1 秒。
- 2026-10-24T02:13:26+08:00 [scheduler] cat_als16_32_200000_d8 完成，耗时 178.6 秒。
- 2026-10-24T02:13:27+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T02:13:27+08:00 [scheduler] 本轮 Go 调度结束；所有日志、配置、检索依据和指标已保存。
- 2026-10-24T02:14:43+08:00 [experiment] GPU 辅助反馈实验 sgns16_n8_e80_s42：设备 mps，AUC 0.752524，最佳 epoch 56，耗时 158.0 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:14:44+08:00 [scheduler] sgns16_n8_e80_s42 完成，耗时 160.8 秒。
- 2026-10-24T02:14:45+08:00 [retrieval] sgns32_n8_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.772891. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:16:23+08:00 [scheduler] Go 调度启动 PID=24874，通道 cpu，每通道最大新实验 15，预算 2.00 小时。
- 2026-10-24T02:16:24+08:00 [retrieval] cat_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.772891. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:16:33+08:00 [progress] 构建 2025-01-24 严格前向快照：392,831 个事件、725,399 个去重用户物品交互、15,164,047 个共点击非零项，耗时 6.5 秒。
- 2026-10-24T02:18:36+08:00 [experiment] GPU 辅助反馈实验 sgns32_n8_e80_s42：设备 mps，AUC 0.710757，最佳 epoch 80，耗时 229.7 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:18:37+08:00 [scheduler] sgns32_n8_e80_s42 完成，耗时 232.2 秒。
- 2026-10-24T02:18:39+08:00 [retrieval] sgns8_n8_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.772891. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:20:21+08:00 [experiment] 实验 cat_context_24_180000_d6：catboost_numeric，切点 2025-01-26，AUC 0.775427，Logloss 0.203946，耗时 234.9 秒。
- 2026-10-24T02:20:21+08:00 [scheduler] cat_context_24_180000_d6 完成，耗时 237.6 秒。
- 2026-10-24T02:20:23+08:00 [retrieval] cat_sgns_context_22_80000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.775427. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:21:02+08:00 [research] 特征模型提升至 AUC 0.772891，最重要信号为 ALS16 归一化相对匹配、历史语义相似度和用户倾向。新增真实历史点击按用户/物品/分钟去重：切点26前946,227次；后续测试频次与时间上下文，全部严格来自拟合期。
- 2026-10-24T02:22:14+08:00 [experiment] GPU 辅助反馈实验 sgns8_n8_e80_s42：设备 mps，AUC 0.783731，最佳 epoch 80，耗时 212.6 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:22:15+08:00 [scheduler] sgns8_n8_e80_s42 完成，耗时 215.9 秒。
- 2026-10-24T02:22:17+08:00 [retrieval] sgns16_n16_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.775427. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:23:56+08:00 [experiment] 实验 cat_sgns_context_22_80000_d6：catboost_numeric，切点 2025-01-26，AUC 0.789505，Logloss 0.200495，耗时 210.4 秒。
- 2026-10-24T02:23:56+08:00 [scheduler] cat_sgns_context_22_80000_d6 完成，耗时 213.4 秒。
- 2026-10-24T02:23:58+08:00 [retrieval] cat_sgns8_16_context_22_80000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.789505. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:24:40+08:00 [experiment] GPU 8维表示在第80轮达到 AUC 0.783731，低于16/32维的容量在这份稀疏数据上更有效。锁定该配置与打分配方，在1月25日切点独立复核；不按复核集重新挑选epoch或打分变体。
- 2026-10-24T02:24:54+08:00 [progress] 构建 2025-01-25 严格前向快照：490,420 个事件、784,246 个去重用户物品交互、18,012,357 个共点击非零项，耗时 10.6 秒。
- 2026-10-24T02:24:59+08:00 [experiment] GPU 辅助反馈实验 sgns16_n16_e80_s42：设备 mps，AUC 0.756626，最佳 epoch 43，耗时 159.9 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:25:00+08:00 [scheduler] sgns16_n16_e80_s42 完成，耗时 163.3 秒。
- 2026-10-24T02:25:02+08:00 [retrieval] sgns8_n16_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.789505. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:27:44+08:00 [experiment] 实验 cat_sgns8_16_context_22_80000_d6：catboost_numeric，切点 2025-01-26，AUC 0.801327，Logloss 0.197110，耗时 224.2 秒。
- 2026-10-24T02:27:45+08:00 [scheduler] cat_sgns8_16_context_22_80000_d6 完成，耗时 227.3 秒。
- 2026-10-24T02:27:47+08:00 [retrieval] cat_sgns_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.801327. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:29:29+08:00 [experiment] GPU 辅助反馈实验 sgns8_n16_e80_s42：设备 mps，AUC 0.774388，最佳 epoch 80，耗时 263.3 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:29:29+08:00 [scheduler] sgns8_n16_e80_s42 完成，耗时 267.4 秒。
- 2026-10-24T02:29:32+08:00 [retrieval] sgns8_n8_e80_s2026: Current best ALS AUC=0.677857; numeric tree AUC=0.801327. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:30:10+08:00 [experiment] GPU 辅助反馈实验 sgns8_locked_cut25：设备 mps，AUC 0.763408，最佳 epoch 80，耗时 327.4 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:33:42+08:00 [experiment] 实验 cat_sgns_context_24_180000_d6：catboost_numeric，切点 2025-01-26，AUC 0.794402，Logloss 0.198748，耗时 352.8 秒。
- 2026-10-24T02:33:43+08:00 [scheduler] cat_sgns_context_24_180000_d6 完成，耗时 356.3 秒。
- 2026-10-24T02:33:44+08:00 [retrieval] cat_sgns8_16_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.801327. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:33:54+08:00 [experiment] GPU 辅助反馈实验 sgns8_n8_e80_s2026：设备 mps，AUC 0.776602，最佳 epoch 80，耗时 258.8 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:33:54+08:00 [scheduler] sgns8_n8_e80_s2026 完成，耗时 262.7 秒。
- 2026-10-24T02:33:56+08:00 [retrieval] sgns8_distinct_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.801327. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:38:23+08:00 [experiment] GPU 辅助反馈实验 sgns8_distinct_e80_s42：设备 mps，AUC 0.785209，最佳 epoch 80，耗时 265.5 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:38:24+08:00 [scheduler] sgns8_distinct_e80_s42 完成，耗时 268.3 秒。
- 2026-10-24T02:38:26+08:00 [retrieval] sgns8_const_e160_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.801327. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:39:37+08:00 [experiment] 实验 cat_sgns8_16_context_24_180000_d6：catboost_numeric，切点 2025-01-26，AUC 0.811689，Logloss 0.193953，耗时 350.0 秒。
- 2026-10-24T02:39:37+08:00 [scheduler] cat_sgns8_16_context_24_180000_d6 完成，耗时 353.1 秒。
- 2026-10-24T02:39:39+08:00 [retrieval] event_cat_sgns8_16_context_22_80000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.811689. Test event capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:39:53+08:00 [experiment] 事件分解实验 event_cat_sgns8_16_context_22_80000_d6：AUC 0.804101，基础模型 0.801327，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T02:39:53+08:00 [scheduler] event_cat_sgns8_16_context_22_80000_d6 完成，耗时 14.8 秒。
- 2026-10-24T02:39:55+08:00 [retrieval] event_cat_sgns8_16_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.811689. Test event capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:40:19+08:00 [experiment] 事件分解实验 event_cat_sgns8_16_context_24_180000_d6：AUC 0.814354，基础模型 0.811689，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T02:40:19+08:00 [scheduler] event_cat_sgns8_16_context_24_180000_d6 完成，耗时 24.5 秒。
- 2026-10-24T02:40:20+08:00 [retrieval] cat_sgns_context_24_180000_d8: Current best ALS AUC=0.677857; numeric tree AUC=0.814354. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:41:21+08:00 [verification] 已将各代 SGNS 特征的训练轮数显式写入配置，区分早期8维56轮与后续80轮；验证重算特征应逐项匹配缓存，防止迭代代码变化导致全量重训与验证配方不一致。
- 2026-10-24T02:43:49+08:00 [research] 数据诊断：主验证历史平均15.29条，评测集13.38条；第二切点14.04条更接近。新增仅用无标签历史长度与已见用户比例的重加权诊断，配方选择结合两切点表现，官方测试指标仍不可知。
- 2026-10-24T02:44:01+08:00 [progress] 构建 2025-01-23 严格前向快照：295,114 个事件、663,037 个去重用户物品交互、12,580,864 个共点击非零项，耗时 8.5 秒。
- 2026-10-24T02:46:31+08:00 [experiment] 实验 cat_sgns_context_24_180000_d8：catboost_numeric，切点 2025-01-26，AUC 0.794522，Logloss 0.198822，耗时 368.6 秒。
- 2026-10-24T02:46:32+08:00 [scheduler] cat_sgns_context_24_180000_d8 完成，耗时 371.5 秒。
- 2026-10-24T02:46:34+08:00 [retrieval] cat_sgns8e80_16_context_22_80000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.814354. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:47:38+08:00 [experiment] GPU 辅助反馈实验 sgns8_const_e160_s42：设备 mps，AUC 0.795586，最佳 epoch 159，耗时 549.5 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:47:39+08:00 [scheduler] sgns8_const_e160_s42 完成，耗时 553.4 秒。
- 2026-10-24T02:47:39+08:00 [scheduler] 本轮 Go 调度结束；所有日志、配置、检索依据和指标已保存。
- 2026-10-24T02:49:24+08:00 [optimization] 较长的8维负采样模型在159轮达到 AUC 0.795586；降低验证频率到每5轮以减少重复排序开销，将更多GPU时间用于参数学习。保留现有最佳检查点与旧模型配方。
- 2026-10-24T02:49:24+08:00 [scheduler] Go 调度启动 PID=44878，通道 gpu，每通道最大新实验 15，预算 2.00 小时。
- 2026-10-24T02:49:28+08:00 [retrieval] sgns8_const_n8_binary_e320_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.814354. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:51:19+08:00 [experiment] 实验 cat_sgns8e80_16_context_22_80000_d6：catboost_numeric，切点 2025-01-26，AUC 0.802837，Logloss 0.196706，耗时 281.1 秒。
- 2026-10-24T02:51:20+08:00 [scheduler] cat_sgns8e80_16_context_22_80000_d6 完成，耗时 286.4 秒。
- 2026-10-24T02:51:23+08:00 [retrieval] cat_sgns8_16_context_24_180000_d8: Current best ALS AUC=0.677857; numeric tree AUC=0.814354. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T02:58:01+08:00 [experiment] 实验 cat_sgns8_16_context_24_180000_d6_locked25：catboost_numeric，切点 2025-01-25，AUC 0.800494，Logloss 0.197801，耗时 847.6 秒。
- 2026-10-24T02:59:47+08:00 [experiment] GPU 辅助反馈实验 sgns8_const_n8_binary_e320_s42：设备 mps，AUC 0.802547，最佳 epoch 320，耗时 614.6 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T02:59:47+08:00 [scheduler] sgns8_const_n8_binary_e320_s42 完成，耗时 619.0 秒。
- 2026-10-24T02:59:49+08:00 [retrieval] sgns8_const_n4_binary_e160_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.814354. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:00:12+08:00 [experiment] 实验 cat_sgns8_16_context_24_180000_d8：catboost_numeric，切点 2025-01-26，AUC 0.811700，Logloss 0.194019，耗时 524.9 秒。
- 2026-10-24T03:00:13+08:00 [scheduler] cat_sgns8_16_context_24_180000_d8 完成，耗时 529.8 秒。
- 2026-10-24T03:00:14+08:00 [retrieval] cat_sgns8long_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.814354. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:03:10+08:00 [experiment] GPU 辅助反馈实验 sgns8_const_n4_binary_e160_s42：设备 mps，AUC 0.780315，最佳 epoch 160，耗时 197.7 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T03:03:11+08:00 [scheduler] sgns8_const_n4_binary_e160_s42 完成，耗时 201.8 秒。
- 2026-10-24T03:03:12+08:00 [experiment] 真实曝光微调 warm_ctr_anchor01：AUC 0.802447，anchor=0.1，最佳epoch 1。
- 2026-10-24T03:03:14+08:00 [retrieval] sgns8_const_n8_distinct_clicks_e160_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.814354. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:07:28+08:00 [experiment] 实验 cat_sgns8long_context_24_180000_d6：catboost_numeric，切点 2025-01-26，AUC 0.819746，Logloss 0.191664，耗时 431.4 秒。
- 2026-10-24T03:07:28+08:00 [scheduler] cat_sgns8long_context_24_180000_d6 完成，耗时 434.3 秒。
- 2026-10-24T03:07:29+08:00 [retrieval] cat_sgns8long_context_24_180000_d8: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:07:41+08:00 [experiment] GPU 辅助反馈实验 sgns8_const_n8_distinct_clicks_e160_s42：设备 mps，AUC 0.787164，最佳 epoch 160，耗时 264.0 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T03:07:42+08:00 [scheduler] sgns8_const_n8_distinct_clicks_e160_s42 完成，耗时 268.1 秒。
- 2026-10-24T03:07:44+08:00 [retrieval] mlp_event_ce_cat_sgns8_16_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:08:58+08:00 [experiment] 前向特征神经评分 mlp_event_ce_cat_sgns8_16_context_24_180000_d6：event_ce，AUC 0.815565，最佳epoch 10。
- 2026-10-24T03:08:59+08:00 [scheduler] mlp_event_ce_cat_sgns8_16_context_24_180000_d6 完成，耗时 75.3 秒。
- 2026-10-24T03:09:01+08:00 [retrieval] mlp_bce_cat_sgns8_16_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:10:55+08:00 [experiment] 前向特征神经评分 mlp_bce_cat_sgns8_16_context_24_180000_d6：bce，AUC 0.812872，最佳epoch 26。
- 2026-10-24T03:10:55+08:00 [scheduler] mlp_bce_cat_sgns8_16_context_24_180000_d6 完成，耗时 113.9 秒。
- 2026-10-24T03:10:57+08:00 [retrieval] mlp_event_ce_cat_sgns8long_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:11:23+08:00 [verification] 验证了主模型200个验证事件的全部100维特征逐项复现，最大误差0。已有11项检查通过，Go vet通过。全量训练采用分块特征缓存，最终模型目录仅保留模型、配方、指标和CSV，避免混入大训练矩阵。
- 2026-10-24T03:11:27+08:00 [experiment] 实验 cat_sgns8long_context_24_180000_d8：catboost_numeric，切点 2025-01-26，AUC 0.819593，Logloss 0.191675，耗时 235.3 秒。
- 2026-10-24T03:11:28+08:00 [scheduler] cat_sgns8long_context_24_180000_d8 完成，耗时 238.3 秒。
- 2026-10-24T03:11:29+08:00 [retrieval] cat_rolling24_sgns8_16_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:11:53+08:00 [experiment] 前向特征神经评分 mlp_event_ce_cat_sgns8long_context_24_180000_d6：event_ce，AUC 0.823270，最佳epoch 10。
- 2026-10-24T03:11:54+08:00 [scheduler] mlp_event_ce_cat_sgns8long_context_24_180000_d6 完成，耗时 56.5 秒。
- 2026-10-24T03:11:56+08:00 [retrieval] mlp_bce_cat_sgns8long_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:13:48+08:00 [experiment] 前向特征神经评分 mlp_bce_cat_sgns8long_context_24_180000_d6：bce，AUC 0.820638，最佳epoch 28。
- 2026-10-24T03:13:48+08:00 [scheduler] mlp_bce_cat_sgns8long_context_24_180000_d6 完成，耗时 112.3 秒。
- 2026-10-24T03:13:51+08:00 [retrieval] sgns6_n8_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:15:56+08:00 [experiment] GPU 辅助反馈实验 sgns6_n8_e80_s42：设备 mps，AUC 0.744271，最佳 epoch 80，耗时 122.7 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T03:15:57+08:00 [scheduler] sgns6_n8_e80_s42 完成，耗时 126.8 秒。
- 2026-10-24T03:15:59+08:00 [retrieval] mlp_event_long_ema_s2026_d0.1: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:17:18+08:00 [experiment] 前向特征神经评分 mlp_event_long_ema_s2026_d0.1：event_ce，AUC 0.823496，最佳epoch 12。
- 2026-10-24T03:17:19+08:00 [scheduler] mlp_event_long_ema_s2026_d0.1 完成，耗时 79.1 秒。
- 2026-10-24T03:17:21+08:00 [retrieval] mlp_event_long_ema_s7_d0.2: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:17:37+08:00 [experiment] 实验 cat_rolling24_sgns8_16_d6：catboost_numeric，切点 2025-01-26，AUC 0.812605，Logloss 0.193630，耗时 365.2 秒。
- 2026-10-24T03:17:37+08:00 [scheduler] cat_rolling24_sgns8_16_d6 完成，耗时 368.3 秒。
- 2026-10-24T03:17:38+08:00 [retrieval] event_cat_sgns8_16_context_24_180000_d8: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test event capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:17:41+08:00 [experiment] 真实曝光向量微调首轮 AUC 0.802447，随后下降，自动早停。改用严格前向快照的固定数值特征训练小神经头：BCE达0.820638，显式11类事件交叉熵达0.823270。所有未点击事件保留，未额外输出未点击行。
- 2026-10-24T03:18:07+08:00 [experiment] 事件分解实验 event_cat_sgns8_16_context_24_180000_d8：AUC 0.814242，基础模型 0.811700，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T03:18:07+08:00 [scheduler] event_cat_sgns8_16_context_24_180000_d8 完成，耗时 28.7 秒。
- 2026-10-24T03:18:08+08:00 [retrieval] cat_sgns8long_context_24_180000_d6_locked25: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:18:25+08:00 [experiment] 前向特征神经评分 mlp_event_long_ema_s7_d0.2：event_ce，AUC 0.823352，最佳epoch 14。
- 2026-10-24T03:18:26+08:00 [scheduler] mlp_event_long_ema_s7_d0.2 完成，耗时 65.3 秒。
- 2026-10-24T03:18:28+08:00 [retrieval] mlp_event_long_ema_s42_d0.2: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:19:14+08:00 [experiment] 前向特征神经评分 mlp_event_long_ema_s42_d0.2：event_ce，AUC 0.823224，最佳epoch 12。
- 2026-10-24T03:19:14+08:00 [scheduler] mlp_event_long_ema_s42_d0.2 完成，耗时 46.8 秒。
- 2026-10-24T03:19:16+08:00 [retrieval] sgns6_const_n8_binary_e160_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:22:27+08:00 [experiment] GPU 辅助反馈实验 sgns6_const_n8_binary_e160_s42：设备 mps，AUC 0.755133，最佳 epoch 160，耗时 189.0 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T03:22:28+08:00 [scheduler] sgns6_const_n8_binary_e160_s42 完成，耗时 192.2 秒。
- 2026-10-24T03:22:30+08:00 [retrieval] sgns4_n8_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:24:31+08:00 [experiment] GPU 辅助反馈实验 sgns4_n8_e80_s42：设备 mps，AUC 0.697389，最佳 epoch 80，耗时 118.0 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T03:24:32+08:00 [scheduler] sgns4_n8_e80_s42 完成，耗时 121.9 秒。
- 2026-10-24T03:24:34+08:00 [retrieval] sgns12_n8_e80_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:26:18+08:00 [experiment] 实验 cat_sgns8long_context_24_180000_d6_locked25：catboost_numeric，切点 2025-01-25，AUC 0.804098，Logloss 0.196867，耗时 487.7 秒。
- 2026-10-24T03:26:19+08:00 [scheduler] cat_sgns8long_context_24_180000_d6_locked25 完成，耗时 490.7 秒。
- 2026-10-24T03:26:19+08:00 [scheduler] 本轮 Go 调度结束；所有日志、配置、检索依据和指标已保存。
- 2026-10-24T03:26:35+08:00 [experiment] GPU 辅助反馈实验 sgns12_n8_e80_s42：设备 mps，AUC 0.777752，最佳 epoch 80，耗时 118.2 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T03:26:36+08:00 [scheduler] sgns12_n8_e80_s42 完成，耗时 121.6 秒。
- 2026-10-24T03:26:37+08:00 [retrieval] mlp_event_ce_cat_sgns8long_context_24_180000_d6_locked25: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:27:02+08:00 [experiment] 前向特征神经评分 mlp_event_ce_cat_sgns8long_context_24_180000_d6_locked25：event_ce，AUC 0.807945，最佳epoch 10。
- 2026-10-24T03:27:02+08:00 [scheduler] mlp_event_ce_cat_sgns8long_context_24_180000_d6_locked25 完成，耗时 25.6 秒。
- 2026-10-24T03:27:02+08:00 [scheduler] 本轮 Go 调度结束；所有日志、配置、检索依据和指标已保存。
- 2026-10-24T03:31:53+08:00 [scheduler] Go 调度启动 PID=72405，通道 all，每通道最大新实验 15，预算 1.50 小时。
- 2026-10-24T03:31:54+08:00 [retrieval] event_cat_sgns8long_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test event capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:31:55+08:00 [retrieval] mlp_event_long_constzero: Current best ALS AUC=0.677857; numeric tree AUC=0.819746. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:32:17+08:00 [experiment] 事件分解实验 event_cat_sgns8long_context_24_180000_d6：AUC 0.822218，基础模型 0.819746，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T03:32:17+08:00 [scheduler] event_cat_sgns8long_context_24_180000_d6 完成，耗时 22.8 秒。
- 2026-10-24T03:32:18+08:00 [retrieval] event_cat_sgns8long_context_24_180000_d8: Current best ALS AUC=0.677857; numeric tree AUC=0.822218. Test event capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:32:39+08:00 [experiment] 事件分解实验 event_cat_sgns8long_context_24_180000_d8：AUC 0.822134，基础模型 0.819593，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T03:32:39+08:00 [scheduler] event_cat_sgns8long_context_24_180000_d8 完成，耗时 20.9 秒。
- 2026-10-24T03:32:40+08:00 [retrieval] event_cat_rolling24_sgns8_16_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.822218. Test event capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:32:43+08:00 [experiment] 前向特征神经评分 mlp_event_long_constzero：event_ce，AUC 0.823270，最佳epoch 10。
- 2026-10-24T03:32:43+08:00 [scheduler] mlp_event_long_constzero 完成，耗时 48.0 秒。
- 2026-10-24T03:32:45+08:00 [retrieval] sgns12_const_n8_binary_e160_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.822218. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:33:15+08:00 [experiment] 事件分解实验 event_cat_rolling24_sgns8_16_d6：AUC 0.815174，基础模型 0.812605，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T03:33:15+08:00 [scheduler] event_cat_rolling24_sgns8_16_d6 完成，耗时 34.7 秒。
- 2026-10-24T03:33:16+08:00 [retrieval] event_cat_sgns8_16_context_24_180000_d6_locked25: Current best ALS AUC=0.677857; numeric tree AUC=0.822218. Test event capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:33:28+08:00 [audit] 13 项单元检查通过。第二时间切点锁定复核：长训练表示 + 事件神经头 AUC 0.807945，对应主切点0.823270。继续比较320轮表示与已知用户历史折入；延长8维辅助表示训练作为有依据的收敛实验，并用小规模固定配方融合降低种子波动。
- 2026-10-24T03:33:33+08:00 [experiment] 事件分解实验 event_cat_sgns8_16_context_24_180000_d6_locked25：AUC 0.802714，基础模型 0.800494，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T03:33:33+08:00 [scheduler] event_cat_sgns8_16_context_24_180000_d6_locked25 完成，耗时 17.0 秒。
- 2026-10-24T03:33:35+08:00 [retrieval] cat_sgns8e320_context24_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.822218. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:34:46+08:00 [experiment] GPU 辅助反馈实验 sgns12_const_n8_binary_e160_s42：设备 mps，AUC 0.773477，最佳 epoch 75，耗时 119.6 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T03:34:47+08:00 [scheduler] sgns12_const_n8_binary_e160_s42 完成，耗时 122.2 秒。
- 2026-10-24T03:34:48+08:00 [retrieval] sgns8_const_n8_binary_e640_s42: Current best ALS AUC=0.677857; numeric tree AUC=0.822218. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:40:11+08:00 [engineering] 补充候选集合的上下文条件化神经头，维持候选置换等变性；新增相应不变量检查。14项测试通过。最终版本将同时保存评分头与其学习到的表示，验收使用独立特征缓存重新生成完整预测。
- 2026-10-24T03:40:58+08:00 [experiment] 实验 cat_sgns8e320_context24_d6：catboost_numeric，切点 2025-01-26，AUC 0.824595，Logloss 0.190208，耗时 441.0 秒。
- 2026-10-24T03:40:59+08:00 [scheduler] cat_sgns8e320_context24_d6 完成，耗时 444.1 秒。
- 2026-10-24T03:41:00+08:00 [retrieval] cat_rolling24_sgns8_16_d8: Current best ALS AUC=0.677857; numeric tree AUC=0.824595. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:42:48+08:00 [decision] 320轮表示的树头 AUC 0.824595，优于159轮的0.819746。根据检索记忆，将剩余种子预算集中到8维长训练；剪枝已显著落后的16/32维种子和无SGNS旧树头，保留剪枝配置与理由。
- 2026-10-24T03:43:58+08:00 [engineering] Go入口已补齐跨通道依赖排空：GPU表示完成可解锁CPU评分头，再解锁GPU神经头；总实验上限和时间预算保持有效。新调度器等待现有任务结束后接续，随后自动执行冻结复核、全量重训、导出及独立验收。
- 2026-10-24T03:43:58+08:00 [scheduler] Go 调度启动 PID=79786，通道 all，每通道最大新实验 15，预算 1.50 小时。
- 2026-10-24T03:43:58+08:00 [scheduler] gpu 通道已有存活调度进程；等待其结束后恢复剩余队列，避免重复训练。
- 2026-10-24T03:43:58+08:00 [scheduler] cpu 通道已有存活调度进程；等待其结束后恢复剩余队列，避免重复训练。
- 2026-10-24T03:45:33+08:00 [acceptance] 再次执行当前轻量入口 light.py --name delivery_smoke：AUC 0.550995→0.581668；2轮训练、最佳模型重载、529390行完整评测预测与CSV检查通过。与重训练并发时全链路18.29秒，结果独立保存在runs/light/delivery_smoke。
- 2026-10-24T03:46:34+08:00 [experiment] GPU 辅助反馈实验 sgns8_const_n8_binary_e640_s42：设备 mps，AUC 0.803856，最佳 epoch 450，耗时 704.1 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T03:46:35+08:00 [scheduler] sgns8_const_n8_binary_e640_s42 完成，耗时 706.7 秒。
- 2026-10-24T03:46:37+08:00 [retrieval] mlp_event_cat_sgns8e320_context24_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.824595. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:47:26+08:00 [experiment] 实验 cat_rolling24_sgns8_16_d8：catboost_numeric，切点 2025-01-26，AUC 0.812830，Logloss 0.193645，耗时 383.4 秒。
- 2026-10-24T03:47:27+08:00 [scheduler] cat_rolling24_sgns8_16_d8 完成，耗时 386.5 秒。
- 2026-10-24T03:47:28+08:00 [retrieval] event_cat_rolling24_sgns8_16_d8: Current best ALS AUC=0.677857; numeric tree AUC=0.824595. Test event capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:47:36+08:00 [experiment] 前向特征神经评分 mlp_event_cat_sgns8e320_context24_d6：event_ce，AUC 0.827889，最佳epoch 8。
- 2026-10-24T03:47:36+08:00 [scheduler] mlp_event_cat_sgns8e320_context24_d6 完成，耗时 59.7 秒。
- 2026-10-24T03:47:38+08:00 [retrieval] mlp_context_cat_sgns8long_context_24_180000_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.824595. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:48:05+08:00 [experiment] 事件分解实验 event_cat_rolling24_sgns8_16_d8：AUC 0.815408，基础模型 0.812830，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T03:48:05+08:00 [scheduler] event_cat_rolling24_sgns8_16_d8 完成，耗时 37.5 秒。
- 2026-10-24T03:48:07+08:00 [retrieval] cat_sgns8e320_context24_d6_locked25: Current best ALS AUC=0.677857; numeric tree AUC=0.824595. Test catboost capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:48:37+08:00 [experiment] 前向特征神经评分 mlp_context_cat_sgns8long_context_24_180000_d6：event_ce，AUC 0.823675，最佳epoch 10。
- 2026-10-24T03:48:38+08:00 [scheduler] mlp_context_cat_sgns8long_context_24_180000_d6 完成，耗时 59.4 秒。
- 2026-10-24T03:48:40+08:00 [retrieval] mlp_context_cat_sgns8long_context_24_180000_d6_locked25: Current best ALS AUC=0.677857; numeric tree AUC=0.824595. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:48:43+08:00 [audit] 对目前领先的320轮树头、320轮事件神经头、159轮EMA神经头及第二切点基准树执行原生模型重载与抽样训练／验证特征重算；所有比较通过。正式验收还会独立重新生成全量评测特征。
- 2026-10-24T03:49:16+08:00 [experiment] 前向特征神经评分 mlp_context_cat_sgns8long_context_24_180000_d6_locked25：event_ce，AUC 0.808980，最佳epoch 10。
- 2026-10-24T03:49:17+08:00 [scheduler] mlp_context_cat_sgns8long_context_24_180000_d6_locked25 完成，耗时 37.1 秒。
- 2026-10-24T03:49:19+08:00 [retrieval] mlp_event_long_ema_s2026_d0.1_locked25: Current best ALS AUC=0.677857; numeric tree AUC=0.824595. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:49:52+08:00 [experiment] 前向特征神经评分 mlp_event_long_ema_s2026_d0.1_locked25：event_ce，AUC 0.808726，最佳epoch 12。
- 2026-10-24T03:49:53+08:00 [scheduler] mlp_event_long_ema_s2026_d0.1_locked25 完成，耗时 34.4 秒。
- 2026-10-24T03:49:55+08:00 [retrieval] mlp_context_cat_sgns8e320_context24_d6: Current best ALS AUC=0.677857; numeric tree AUC=0.824595. Test tabular capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:50:45+08:00 [experiment] 前向特征神经评分 mlp_context_cat_sgns8e320_context24_d6：event_ce，AUC 0.827841，最佳epoch 10。
- 2026-10-24T03:50:46+08:00 [scheduler] mlp_context_cat_sgns8e320_context24_d6 完成，耗时 51.2 秒。
- 2026-10-24T03:50:48+08:00 [retrieval] sgns8_const_e320_s2026: Current best ALS AUC=0.677857; numeric tree AUC=0.824595. Test sgns capacity/confidence using retrieved evidence; keep temporal split fixed.
- 2026-10-24T03:52:47+08:00 [experiment] 补齐轻量版数据／模型对照，同一4000用户验证集：无历史0.551171、最近5次历史0.570221、最近20次历史8维0.581668、20次历史16维0.582479。比较入口与产物已写入LIGHT_START.md。小样本增益不替代完整时间验证。
- 2026-10-24T03:56:02+08:00 [experiment] GPU 辅助反馈实验 sgns8_const_e320_s2026：设备 mps，AUC 0.802377，最佳 epoch 320，耗时 311.4 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。
- 2026-10-24T03:56:02+08:00 [scheduler] sgns8_const_e320_s2026 完成，耗时 314.7 秒。
- 2026-10-24T03:56:04+08:00 [pruned] sgns16_n8_e80_s2026: 8-dimensional representation exceeded AUC 0.80; completed 16/32-dimensional runs were materially weaker. Reallocate seed budget to the stronger low-dimensional family.
- 2026-10-24T03:56:04+08:00 [pruned] sgns32_n8_e80_s2026: 8-dimensional representation exceeded AUC 0.80; completed 16/32-dimensional runs were materially weaker. Reallocate seed budget to the stronger low-dimensional family.
- 2026-10-24T03:56:05+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T03:56:17+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T03:58:30+08:00 [experiment] 实验 cat_sgns8e320_context24_d6_locked25：catboost_numeric，切点 2025-01-25，AUC 0.808800，Logloss 0.195470，耗时 620.3 秒。
- 2026-10-24T03:58:31+08:00 [scheduler] cat_sgns8e320_context24_d6_locked25 完成，耗时 623.5 秒。
- 2026-10-24T03:58:32+08:00 [retrieval] cat_sgns8_s2026_context24_d6: Current best ALS AUC=0.677857; tree-family AUC=0.824595; event neural head AUC=0.827889. Test catboost with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T03:58:34+08:00 [scheduler] Go 调度启动 PID=88191，通道 gpu，每通道最大新实验 15，预算 1.10 小时。
- 2026-10-24T03:58:36+08:00 [retrieval] mlp_event_cat_sgns8e320_context24_d6_locked25: Current best ALS AUC=0.677857; tree-family AUC=0.824595; event neural head AUC=0.827889. Test tabular with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T03:59:01+08:00 [experiment] 前向特征神经评分 mlp_event_cat_sgns8e320_context24_d6_locked25：event_ce，AUC 0.812416，最佳epoch 8。
- 2026-10-24T03:59:02+08:00 [scheduler] mlp_event_cat_sgns8e320_context24_d6_locked25 完成，耗时 25.9 秒。
- 2026-10-24T03:59:04+08:00 [retrieval] mlp_context_cat_sgns8e320_context24_d6_locked25: Current best ALS AUC=0.677857; tree-family AUC=0.824595; event neural head AUC=0.827889. Test tabular with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T03:59:38+08:00 [experiment] 前向特征神经评分 mlp_context_cat_sgns8e320_context24_d6_locked25：event_ce，AUC 0.812771，最佳epoch 10。
- 2026-10-24T03:59:39+08:00 [scheduler] mlp_context_cat_sgns8e320_context24_d6_locked25 完成，耗时 35.3 秒。
- 2026-10-24T04:00:42+08:00 [audit] 轻量演示增加按完整用户事件配对的200次bootstrap：双塔相对CTR增益区间与16维对8维增益区间保存在runs/light/comparison_delivery.json。该区间对固定已选配方计算，不含早停选择不确定性。
- 2026-10-24T04:02:45+08:00 [resources] 利用GPU等待CPU评分头的空档，预先训练三个已冻结候选在Jan27完整训练截止点的辅助表示；不读取评测标签、不产生或选择验证分数。最终重训可直接复用相同参数的表示缓存。
- 2026-10-24T04:03:01+08:00 [progress] 构建 2025-01-27 严格前向快照：686,824 个事件、896,564 个去重用户物品交互、24,482,597 个共点击非零项，耗时 15.9 秒。
- 2026-10-24T04:04:40+08:00 [experiment] 实验 cat_sgns8_s2026_context24_d6：catboost_numeric，切点 2025-01-26，AUC 0.824114，Logloss 0.190323，耗时 366.4 秒。
- 2026-10-24T04:04:41+08:00 [scheduler] cat_sgns8_s2026_context24_d6 完成，耗时 369.5 秒。
- 2026-10-24T04:04:42+08:00 [retrieval] mlp_event_cat_sgns8_s2026_context24_d6: Current best ALS AUC=0.677857; tree-family AUC=0.824595; event neural head AUC=0.827889. Test tabular with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:04:43+08:00 [retrieval] cat_sgns8_s2026_context24_d6_locked25: Current best ALS AUC=0.677857; tree-family AUC=0.824595; event neural head AUC=0.827889. Test catboost with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:05:42+08:00 [experiment] 前向特征神经评分 mlp_event_cat_sgns8_s2026_context24_d6：event_ce，AUC 0.827855，最佳epoch 10。
- 2026-10-24T04:05:43+08:00 [scheduler] mlp_event_cat_sgns8_s2026_context24_d6 完成，耗时 60.7 秒。
- 2026-10-24T04:08:20+08:00 [resources] 全训练期表示缓存已完成：cat_sgns8e320_context24_d6
- 2026-10-24T04:12:29+08:00 [resources] 全训练期表示缓存已完成：cat_sgns8_s2026_context24_d6
- 2026-10-24T04:13:18+08:00 [experiment] 前向特征神经评分 feature_reference_parity_smoke25：event_ce，AUC 0.807735，最佳epoch 1。
- 2026-10-24T04:13:18+08:00 [audit] 回溯神经头可直接使用冻结特征引用；同数据、种子、首轮训练与原路径的AUC完全一致。Jan24验收采用此路径，缩短中间模型训练时间。
- 2026-10-24T04:15:03+08:00 [experiment] 实验 cat_sgns8_s2026_context24_d6_locked25：catboost_numeric，切点 2025-01-25，AUC 0.812736，Logloss 0.194315，耗时 617.1 秒。
- 2026-10-24T04:15:04+08:00 [scheduler] cat_sgns8_s2026_context24_d6_locked25 完成，耗时 621.2 秒。
- 2026-10-24T04:15:05+08:00 [retrieval] cat_sgns8e640_context24_d6: Current best ALS AUC=0.677857; tree-family AUC=0.824595; event neural head AUC=0.827889. Test catboost with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:15:13+08:00 [retrieval] mlp_event_cat_sgns8_s2026_context24_d6_locked25: Current best ALS AUC=0.677857; tree-family AUC=0.824595; event neural head AUC=0.827889. Test tabular with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:15:57+08:00 [experiment] 前向特征神经评分 mlp_event_cat_sgns8_s2026_context24_d6_locked25：event_ce，AUC 0.815970，最佳epoch 10。
- 2026-10-24T04:15:58+08:00 [scheduler] mlp_event_cat_sgns8_s2026_context24_d6_locked25 完成，耗时 45.6 秒。
- 2026-10-24T04:18:45+08:00 [resources] 全训练期表示缓存已完成：cat_sgns8e640_context24_d6
- 2026-10-24T04:19:23+08:00 [decision] 将已验证有效的事件点击率分解应用到新长训练树头，单次只需训练小型事件门控。最终候选池保留最佳逐行树、最佳事件树和前三个神经头，以两个时间切点的粗权重融合比较互补性。
- 2026-10-24T04:22:08+08:00 [resources] 利用空闲GPU预备Jan24冻结复核需要的Jan22之前表示。只读取Jan22前数据，不计算Jan24分数。
- 2026-10-24T04:24:28+08:00 [experiment] 实验 cat_sgns8e640_context24_d6：catboost_numeric，切点 2025-01-26，AUC 0.825058，Logloss 0.190112，耗时 559.4 秒。
- 2026-10-24T04:24:28+08:00 [scheduler] cat_sgns8e640_context24_d6 完成，耗时 563.0 秒。
- 2026-10-24T04:24:29+08:00 [retrieval] mlp_event_cat_sgns8e640_context24_d6: Current best ALS AUC=0.677857; tree-family AUC=0.825058; event neural head AUC=0.827889. Test tabular with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:24:30+08:00 [retrieval] cat_sgns8e640_context24_d6_locked25: Current best ALS AUC=0.677857; tree-family AUC=0.825058; event neural head AUC=0.827889. Test catboost with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:25:18+08:00 [experiment] 前向特征神经评分 mlp_event_cat_sgns8e640_context24_d6：event_ce，AUC 0.828389，最佳epoch 8。
- 2026-10-24T04:25:19+08:00 [scheduler] mlp_event_cat_sgns8e640_context24_d6 完成，耗时 49.7 秒。
- 2026-10-24T04:25:33+08:00 [resources] Jan24冻结复核的早期表示缓存已完成：cat_sgns8e320_context24_d6
- 2026-10-24T04:27:55+08:00 [resources] Jan24冻结复核的早期表示缓存已完成：cat_sgns8_s2026_context24_d6
- 2026-10-24T04:30:05+08:00 [retrieval] mlp_aligned_s2026_e320: Current best ALS AUC=0.677857; tree-family AUC=0.825058; event neural head AUC=0.828389. Test tabular with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:30:36+08:00 [research] 检索SciPy原始文档并验证正交Procrustes：两快照同物品向量平均对应余弦0.576→0.921，正交误差2.7e-15。增加一轮对齐坐标的事件神经头：锚点取训练特征窗口之前快照；第二与第三切点自动同步前移锚点；最终版本会一同保存锚点表示。是否保留仅由AUC验证决定。
- 2026-10-24T04:30:47+08:00 [experiment] 前向特征神经评分 mlp_aligned_s2026_e320：event_ce，AUC 0.826631，最佳epoch 5。
- 2026-10-24T04:30:48+08:00 [scheduler] mlp_aligned_s2026_e320 完成，耗时 43.1 秒。
- 2026-10-24T04:34:42+08:00 [decision] 对齐坐标事件头主验证AUC0.826631，低于同表示原匹配特征头0.827855；保留研究结果，不进入当前前三神经候选。15项测试通过。停止增加新模型家族，完成现有事件校准／历史折入队列后冻结交付。
- 2026-10-24T04:36:19+08:00 [experiment] 实验 cat_sgns8e640_context24_d6_locked25：catboost_numeric，切点 2025-01-25，AUC 0.813338，Logloss 0.194205，耗时 706.3 秒。
- 2026-10-24T04:36:20+08:00 [scheduler] cat_sgns8e640_context24_d6_locked25 完成，耗时 709.8 秒。
- 2026-10-24T04:36:21+08:00 [retrieval] event_cat_sgns8e320_context24_d6: Current best ALS AUC=0.677857; tree-family AUC=0.825058; event neural head AUC=0.828389. Test event with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:36:22+08:00 [retrieval] mlp_event_cat_sgns8e640_context24_d6_locked25: Current best ALS AUC=0.677857; tree-family AUC=0.825058; event neural head AUC=0.828389. Test tabular with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:36:49+08:00 [experiment] 事件分解实验 event_cat_sgns8e320_context24_d6：AUC 0.827157，基础模型 0.824595，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T04:36:49+08:00 [scheduler] event_cat_sgns8e320_context24_d6 完成，耗时 28.7 秒。
- 2026-10-24T04:36:51+08:00 [retrieval] event_cat_sgns8_s2026_context24_d6: Current best ALS AUC=0.677857; tree-family AUC=0.827157; event neural head AUC=0.828389. Test event with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:36:52+08:00 [experiment] 前向特征神经评分 mlp_event_cat_sgns8e640_context24_d6_locked25：event_ce，AUC 0.816275，最佳epoch 8。
- 2026-10-24T04:36:53+08:00 [scheduler] mlp_event_cat_sgns8e640_context24_d6_locked25 完成，耗时 31.2 秒。
- 2026-10-24T04:37:16+08:00 [experiment] 事件分解实验 event_cat_sgns8_s2026_context24_d6：AUC 0.826707，基础模型 0.824114，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T04:37:16+08:00 [scheduler] event_cat_sgns8_s2026_context24_d6 完成，耗时 25.4 秒。
- 2026-10-24T04:37:17+08:00 [retrieval] event_cat_sgns8e640_context24_d6: Current best ALS AUC=0.677857; tree-family AUC=0.827157; event neural head AUC=0.828389. Test event with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:37:38+08:00 [experiment] 事件分解实验 event_cat_sgns8e640_context24_d6：AUC 0.827592，基础模型 0.825058，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T04:37:38+08:00 [scheduler] event_cat_sgns8e640_context24_d6 完成，耗时 21.2 秒。
- 2026-10-24T04:37:38+08:00 [scheduler] 本轮 Go 调度结束；所有日志、配置、检索依据和指标已保存。
- 2026-10-24T04:37:43+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T04:37:45+08:00 [retrieval] cat_foldin_long24_d6: Current best ALS AUC=0.677857; tree-family AUC=0.827592; event neural head AUC=0.828389. Test catboost with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:42:40+08:00 [experiment] 实验 cat_foldin_long24_d6：catboost_numeric，切点 2025-01-26，AUC 0.819896，Logloss 0.191627，耗时 292.9 秒。
- 2026-10-24T04:42:40+08:00 [scheduler] cat_foldin_long24_d6 完成，耗时 295.4 秒。
- 2026-10-24T04:42:41+08:00 [pruned] cat_context_24_180000_d8: Numeric/ALS-only heads have completed representatives and trail SGNS-enhanced heads by over 0.02 AUC; prioritize stronger representations and locked audits.
- 2026-10-24T04:42:41+08:00 [pruned] cat_context_22_200000_d6: Numeric/ALS-only heads have completed representatives and trail SGNS-enhanced heads by over 0.02 AUC; prioritize stronger representations and locked audits.
- 2026-10-24T04:42:41+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T04:42:42+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T04:42:43+08:00 [retrieval] mlp_event_cat_foldin_long24_d6: Current best ALS AUC=0.677857; tree-family AUC=0.827592; event neural head AUC=0.828389. Test tabular with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:42:47+08:00 [retrieval] mlp_event_cat_foldin_long24_d6: Current best ALS AUC=0.677857; tree-family AUC=0.827592; event neural head AUC=0.828389. Test tabular with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts.
- 2026-10-24T04:43:31+08:00 [experiment] 前向特征神经评分 mlp_event_cat_foldin_long24_d6：event_ce，AUC 0.823455，最佳epoch 12。
- 2026-10-24T04:43:32+08:00 [scheduler] mlp_event_cat_foldin_long24_d6 完成，耗时 48.4 秒。
- 2026-10-24T04:43:33+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T04:43:35+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T04:43:36+08:00 [experiment] 前向特征神经评分 mlp_event_cat_foldin_long24_d6：event_ce，AUC 0.823455，最佳epoch 12。
- 2026-10-24T04:43:36+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T04:43:36+08:00 [scheduler] mlp_event_cat_foldin_long24_d6 完成，耗时 49.3 秒。
- 2026-10-24T04:43:37+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T04:43:38+08:00 [audit] 锁定主验证配置，开始 2025-01-25 时间切点复核 event_cat_sgns8e640_context24_d6。
- 2026-10-24T04:43:39+08:00 [scheduler] Current candidate queue exhausted; review results and add evidence-backed methods before finalization.
- 2026-10-24T04:43:39+08:00 [scheduler] 本轮 Go 调度结束；所有日志、配置、检索依据和指标已保存。
- 2026-10-24T04:43:51+08:00 [experiment] 事件分解实验 event_cat_sgns8e640_context24_d6_locked25：AUC 0.815512，基础模型 0.813338，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T04:44:25+08:00 [audit] 锁定主验证配置，开始 2025-01-24 时间切点复核 event_cat_sgns8e640_context24_d6。
- 2026-10-24T04:44:25+08:00 [audit] 锁定主验证配置，开始 2025-01-24 时间切点复核 cat_sgns8e640_context24_d6。
- 2026-10-24T04:48:18+08:00 [audit] 审计发现旧版运行进程不识别新版waiting状态，折入神经头重复运行一次；两次AUC均0.823455，未入选。当前Go增加OS级通道排他锁，等待超时与锁释放测试通过；正式验收纳入该测试。冻结配方选中事件树450轮表示与seed2026神经头320轮表示，logit各50%；Jan26 AUC0.829004、Jan25 AUC0.818804。
- 2026-10-24T04:51:06+08:00 [experiment] 实验 cat_sgns8e640_context24_d6_locked24：catboost_numeric，切点 2025-01-24，AUC 0.800866，Logloss 0.196933，耗时 401.2 秒。
- 2026-10-24T04:51:16+08:00 [experiment] 事件分解实验 event_cat_sgns8e640_context24_d6_locked24：AUC 0.802905，基础模型 0.800866，温度 1，事件分支占比 0.75。保留未点击结果。
- 2026-10-24T04:51:16+08:00 [audit] 锁定主验证配置，开始 2025-01-24 时间切点复核 mlp_event_cat_sgns8_s2026_context24_d6。
- 2026-10-24T04:51:54+08:00 [experiment] 前向特征神经评分 mlp_event_cat_sgns8_s2026_context24_d6_locked24：event_ce，AUC 0.802686，最佳epoch 10。
- 2026-10-24T04:51:55+08:00 [audit] 未参与参数和权重选择的 Jan24 回溯复核 AUC 0.804915；冻结配方继续全量重训。
- 2026-10-24T04:51:55+08:00 [finalize] 双切点选择配方：{'kind': 'logit', 'weights': [0.5, 0.5], 'auc_26': 0.8290037149576445, 'auc_25': 0.8188039296003504, 'weighted_auc_26': 0.8206860488965726, 'weighted_auc_25': 0.8157741408544193, 'selection_score': 0.8182300948754959}，来源 ['event_cat_sgns8e640_context24_d6', 'mlp_event_cat_sgns8_s2026_context24_d6']；开始全量重训。
- 2026-10-24T05:00:24+08:00 [audit] 公平基线复核：原单精度优化实现AUC0.615649；按官方starter双精度公式重新计算AUC0.615704。概率接近1的舍入并列造成微小差异，最终提升与置信区间使用双精度参考。
- 2026-10-24T05:00:34+08:00 [delivery] 最终 CSV 导出并检查通过，529,390 行；主验证 AUC 0.829004，第二切点 0.818804。
- 2026-10-24T05:00:35+08:00 [delivery] 最终源码包已补齐Go通道互斥测试文件；正式验收将计算其SHA256。
- 2026-10-24T05:01:16+08:00 [acceptance] 独立验收通过：保存模型和双切点特征重算一致；全部单元测试与Go静态检查通过；529,390行CSV覆盖与概率检查通过；重新加载模型预测最大差异0。
- 2026-10-24T05:01:16+08:00 [scheduler] 本轮 Go 调度结束；所有日志、配置、检索依据和指标已保存。
- 2026-10-24T05:03:24+08:00 [complete] 最终交付收尾完成：100组已完成实验；15项Python检查、Go静态检查与互斥测试通过；65个模型／表示／源码文件哈希核验通过；529390行提交CSV重载复现差异0；全部训练和caffeinate进程已退出。轻量入口、数据／模型对照、最终CSV和模型均可直接使用。
- 2026-10-24T10:47:37+08:00 [v2] 已冻结 v1 当前最优 CSV、清单与验收记录到 competition/champions/v1；原完整模型版本保留。开始 v2 扩大评分网络结构，保持旧提交不动。
- 2026-10-24T10:49:38+08:00 [experiment] 前向特征神经评分 v2_wide256_s2026：event_ce，AUC 0.827959，最佳epoch 8。
- 2026-10-24T10:50:22+08:00 [experiment] 前向特征神经评分 v2_cross256_s2026：event_ce，AUC 0.827848，最佳epoch 5。
- 2026-10-24T10:51:00+08:00 [v2] 本轮清理两份未入选的对齐／fold-in 中间特征缓存以腾出约 2GB；保留其模型、配置与验证预测，v1 所需文件全部保留，缓存可按原配置再生成。
- 2026-10-24T10:52:57+08:00 [experiment] 前向特征神经评分 v2_attention128_s2026：event_ce，AUC 0.827647，最佳epoch 6。
- 2026-10-24T10:55:41+08:00 [experiment] 前向特征神经评分 v2_attention128_s2026：event_ce，AUC 0.827647，最佳epoch 6。
- 2026-10-24T10:56:31+08:00 [experiment] 前向特征神经评分 v2_multiview_wide256_s2026：event_ce，AUC 0.829246，最佳epoch 8。
- 2026-10-24T10:57:15+08:00 [experiment] 前向特征神经评分 v2_multiview_cross256_s2026：event_ce，AUC 0.829103，最佳epoch 5。
- 2026-10-24T10:57:15+08:00 [audit] 锁定主验证配置，开始 2025-01-25 时间切点复核 v2_wide256_s2026。
- 2026-10-24T10:57:44+08:00 [experiment] 前向特征神经评分 v2_wide256_s2026_locked25：event_ce，AUC 0.816084，最佳epoch 8。
- 2026-10-24T10:57:44+08:00 [audit] 锁定主验证配置，开始 2025-01-25 时间切点复核 v2_cross256_s2026。
- 2026-10-24T10:58:05+08:00 [experiment] 前向特征神经评分 v2_cross256_s2026_locked25：event_ce，AUC 0.815776，最佳epoch 5。
- 2026-10-24T10:58:05+08:00 [audit] 锁定主验证配置，开始 2025-01-25 时间切点复核 v2_attention128_s2026。
- 2026-10-24T10:59:26+08:00 [experiment] 前向特征神经评分 v2_attention128_s2026_locked25：event_ce，AUC 0.816357，最佳epoch 6。
- 2026-10-24T10:59:26+08:00 [audit] 锁定主验证配置，开始 2025-01-25 时间切点复核 v2_multiview_wide256_s2026。
- 2026-10-24T10:59:56+08:00 [experiment] 前向特征神经评分 v2_multiview_wide256_s2026_locked25：event_ce，AUC 0.818649，最佳epoch 8。
- 2026-10-24T10:59:56+08:00 [audit] 锁定主验证配置，开始 2025-01-25 时间切点复核 v2_multiview_cross256_s2026。
- 2026-10-24T11:00:17+08:00 [experiment] 前向特征神经评分 v2_multiview_cross256_s2026_locked25：event_ce，AUC 0.818431，最佳epoch 5。
- 2026-10-24T11:00:35+08:00 [v2] v2 固定粗网格比较完成：主切点 0.829327，第二切点 0.819246，满足双切点改进的配方 10 个。
- 2026-10-24T11:00:38+08:00 [audit] 锁定主验证配置，开始 2025-01-24 时间切点复核 v2_multiview_wide256_s2026。
- 2026-10-24T11:01:24+08:00 [experiment] 前向特征神经评分 v2_multiview_wide256_s2026_locked24：event_ce，AUC 0.805364，最佳epoch 8。
- 2026-10-24T11:01:24+08:00 [audit] 锁定主验证配置，开始 2025-01-24 时间切点复核 v2_multiview_cross256_s2026。
- 2026-10-24T11:01:46+08:00 [experiment] 前向特征神经评分 v2_multiview_cross256_s2026_locked24：event_ce，AUC 0.804750，最佳epoch 5。
- 2026-10-24T11:01:47+08:00 [audit] 未参与参数和权重选择的 Jan24 回溯复核 AUC 0.805402；冻结配方继续全量重训。
- 2026-10-24T11:01:47+08:00 [finalize] 双切点选择配方：{'kind': 'logit', 'weights': [0.25, 0.25, 0.25, 0.25], 'auc_26': 0.8293272082809982, 'auc_25': 0.8192463525707806, 'weighted_auc_26': 0.8210333454158742, 'weighted_auc_25': 0.8162117925854054, 'selection_score': 0.8186225690006399}，来源 ['event_cat_sgns8e640_context24_d6', 'mlp_event_cat_sgns8_s2026_context24_d6', 'v2_multiview_wide256_s2026', 'v2_multiview_cross256_s2026']；开始全量重训。
- 2026-10-24T11:03:41+08:00 [delivery] 最终 CSV 导出并检查通过，529,390 行；主验证 AUC 0.829327，第二切点 0.819246。
- 2026-10-24T11:04:42+08:00 [acceptance] 独立验收通过：保存模型和双切点特征重算一致；全部单元测试与Go静态检查通过；529,390行CSV覆盖与概率检查通过；重新加载模型预测最大差异0。
- 2026-10-24T11:04:42+08:00 [v2] v2 结构扩展已完成并独立验收，新版晋升；v1 完整保留。
- 2026-10-24T11:05:41+08:00 [v2] 新版独立验收后清理 3 份可再生成的全量重训特征矩阵，释放 3.42 GiB；新旧模型、全部验证特征和预测文件已保留。
- 2026-10-24T14:17:00+08:00 [generative] 生成式轻量分支 qwen35_softtoken_smoke：Qwen3.5真实训练 24 步，峰值 2.76 GiB，同队列 AUC 0.515393，当前最优提交保留。
- 2026-10-24T17:10:48+08:00 [generative] 生成式轻量分支 qwen35_softtoken_light：Qwen3.5真实训练 128 步，峰值 6.79 GiB，同队列 AUC 0.618767，当前最优提交保留。
- 2026-10-24T18:31:52+08:00 [generative] Qwen3.5 热启动扩大训练进行中：已完成约 5,120 次事件样本，同一 2,048 事件队列 AUC 从 0.581322 提升到 0.731601。冷启动失败记录与权重已保留，原 V2 最优提交未改变。
- 2026-10-24T18:52:57+08:00 [generative] 生成式轻量分支 qwen35_softtoken_8192_warm：Qwen3.5真实训练 1024 步，峰值 5.85 GiB，同队列 AUC 0.756904，当前最优提交保留。
- 2026-10-24T19:24:57+08:00 [generative] 生成式训练与提交完成：qwen35_softtoken_8192_warm，完整 CSV 529,390 行，Jan26 完整时间验证 AUC 0.751811；独立重载验收通过，原最优保留。
- 2026-10-24T23:24:47+08:00 [experiment] 前向特征神经评分 v3_wide512_s73：event_ce，AUC 0.829267，最佳epoch 10。
- 2026-10-24T23:28:38+08:00 [audit] 锁定主验证配置，开始 2025-01-25 时间切点复核 v3_wide512_s73。
- 2026-10-24T23:29:29+08:00 [experiment] 前向特征神经评分 v3_wide512_s73_locked25：event_ce，AUC 0.818701，最佳epoch 10。
- 2026-10-24T23:30:26+08:00 [experiment] 前向特征神经评分 v3_cross512_s2027：event_ce，AUC 0.829132，最佳epoch 4。
- 2026-10-24T23:30:28+08:00 [audit] 锁定主验证配置，开始 2025-01-25 时间切点复核 v3_cross512_s2027。
- 2026-10-24T23:30:58+08:00 [experiment] 前向特征神经评分 v3_cross512_s2027_locked25：event_ce，AUC 0.818591，最佳epoch 4。
- 2026-10-25T01:42:55+08:00 [v3] V3 五轮结构／基模迭代与双切点复核完成，交付 CSV 529,390 行，Jan26 AUC 0.829341，Jan25 AUC 0.819250，原最优保留。
- 2026-10-25T06:06:38+08:00 [progress] 构建 2025-01-21 严格前向快照：98,514 个事件、470,337 个去重用户物品交互、7,498,319 个共点击非零项，耗时 3.0 秒。

## 目录

- `competition/experiments/`：每轮配置、指标、验证预测与模型。
- `competition/cache/`：只按拟合时间构建的稀疏交互、协同矩阵和特征缓存。
- `competition/research/`：资料、检索结果和实验计划。
- `competition/logs/`：Go 调度与训练原始日志。
- `competition/final/`：最终推荐 CSV、模型清单、校验与复现入口。
- `LIGHT_START.md`：之前的轻量链路教程。

全量流程的改造入口：`arena_features.py` 汇总特征，`arena_feedback.py` 处理历史反馈与上下文统计，`arena_sgns.py` 训练辅助表示，`arena_tabular.py` 定义神经评分头，`arena_plan.py` 定义检索和候选实验。每次用新实验名保留配置与结果；只修改 SGNS 训练轮数也会校验缓存配方，避免复用不匹配的特征。
