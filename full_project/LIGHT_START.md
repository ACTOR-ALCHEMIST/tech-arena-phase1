# 轻量实验入门

用同一份小规模时间验证集，依次观察常数预测、平滑用户与商品点击率、两轮小型双塔训练的结果。默认保留 4,000 个验证用户，使用这些用户在切点前的全部事件拟合，并在切点后每人首个事件上验证。

## 运行第一轮

```bash
cd /Users/jasper/Desktop/TechArena2026
python3 light.py
```

当前项目已准备好 `data/cache_v1`。从原始数据重新开始时，先运行一次 `python3 run.py prepare`；这一步需要扫描完整 CSV，后续轻量实验直接读取缓存。

默认配置位于 `configs/light.json`：8 维向量，2 个 epoch，CPU，固定抽样种子。每个事件包含完整的 10 个候选，训练和验证按 UTC 时间分开；所有点击率统计仅使用本轮小训练集标签。

终端会依次显示：

1. 小训练集和验证集的规模。
2. 常数预测与平滑用户／商品点击率的 AUC。
3. 每轮训练后的 AUC，以及相对点击率基线的增量。
4. 最佳模型重新加载后的完整评测预测和 CSV 检查。

当前环境的 `runs/light/baseline_demo` 实测使用 56,455 个训练事件、4,000 个验证事件，数据约 15 MB。两轮训练约 1.7 秒，包含数据读取、模型保存、完整评测预测与检查的链路约 7.1 秒。

| 阶段 | 本地验证 AUC |
|---|---:|
| 常数预测 | 0.500000 |
| 平滑用户／商品点击率 | 0.550995 |
| 第 1 轮训练 | 0.581668 |
| 第 2 轮训练 | 0.580510 |

最佳轮相对点击率基线提升约 0.030673。后续实验会如实记录提升或下降，耗时与指标以本机实际输出为准。

这里的点击率基线是一个独立的简单统计基线。官方 `starter_solution.py` 还包含共点击模块，其结果需要另行评估。

已补跑的数据、模型对照均使用上述同一组 4,000 个验证事件：

| 对照 | 最佳验证 AUC | 结果目录 |
|---|---:|---|
| 不使用历史，8 维 | 0.551171 | `runs/light/no_history_demo/` |
| 最近 5 次点击，8 维 | 0.570221 | `runs/light/history5_demo/` |
| 最近 20 次点击，8 维 | 0.581668 | `runs/light/delivery_smoke/` |
| 最近 20 次点击，16 维 | 0.582479 | `runs/light/dim16_demo/` |

这组小实验显示历史信息有用；16 维的增益很小，需要更多种子和完整时间验证才能判断。`delivery_smoke` 也重新加载了模型并检查了完整 529,390 行评测预测；与大实验并发时全链路约 18 秒。

按完整用户事件配对重采样 200 次，固定 8 维模型相对 CTR 的 AUC 增益 95% 区间约为 `[0.0259, 0.0366]`；16 维相对 8 维约为 `[-0.0013, 0.0029]`。因此当前足以观察到训练相对基线的改善，但还不能确认增大维度更好。完整记录在 `runs/light/comparison_delivery.json`；区间没有计入早停选模的不确定性。

直接比较现有结果：

```bash
python3 light.py --compare runs/light/delivery_smoke runs/light/history5_demo runs/light/dim16_demo runs/light/no_history_demo
```

## 查看实验产物

每次自动创建新的 `runs/light/<时间与随机后缀>/`，也可以用 `--name` 指定名称。已有实验不会被覆盖。

| 文件 | 内容 |
|---|---|
| `config.json` | 本轮实际采用的配置，包括命令行覆盖项 |
| `summary.json` | 数据划分标识、基线、最佳 AUC、增量、分组表现和总耗时 |
| `metrics.csv` | 两个基线和每轮训练的 AUC、Logloss、AUC 增量 |
| `split.npz` | 原始缓存中的训练、验证事件索引 |
| `model.pt` | 最佳模型参数和所需的数据处理设置 |
| `model.validation.npz` | 验证标签、预测、点击率基线预测和原始事件索引 |
| `submission_test.csv` | 覆盖完整评测集的链路测试预测，自动检查表头、概率及行号 |
| `source/` | 本轮使用的代码快照，便于后续修改后回看 |

`submission_test.csv` 来自小训练集模型，主要用于确认链路。比赛成绩选择应结合完整训练期验证与全量重训。

## 做第一个数据实验

先在固定验证集上对比最近 5 次点击与最近 20 次点击：

```bash
python3 light.py --name history5 --history-limit 5 --skip-predict
python3 light.py --name history20 --history-limit 20 --skip-predict
python3 light.py --compare runs/light/history5 runs/light/history20
```

也可以将 `--history-limit` 设为 `0`，同时清空历史物品、间隔、长度、重复点击等历史输入，观察模型是否依赖这些信息。

比较数据量时，使用 `--fit-fraction`，它保持验证集固定：

```bash
python3 light.py --name quarter_data --fit-fraction 0.25 --skip-predict
```

历史裁剪和相关输入特征在 `ctr_data.py` 的 `with_history_limit()`；按用户抽样、时间切分和训练数据量在 `light_data.py` 的 `build_light_dataset()`。

## 做第一个模型实验

固定数据，只改变向量维度：

```bash
python3 light.py --name dim8 --dim 8 --skip-predict
python3 light.py --name dim16 --dim 16 --skip-predict
python3 light.py --compare runs/light/dim8 runs/light/dim16
```

网络位于 `ctr_model.py` 的 `HistoryTwoTower`。配置中的 `use_history_pooling` 控制历史向量分支，`use_repeat_features` 控制重复点击分支。两者独立；彻底移除历史输入时使用 `history_limit=0`。

学习率、正则强度、用户 ID dropout、batch 大小和 epoch 数都可在 `configs/light.json` 中修改。用 `--config` 指向复制后的 JSON 可保留多组配置。

## 公平比较结果

保持 `cutoff`、`cohort_users`、`split_seed` 不变，验证事件就保持一致。模型的 `seed` 单独控制参数初始化和训练顺序，不会重新抽取用户。

`validation_id` 相同表示验证行相同；`split_id` 还包含训练事件，训练数据量变化时会变化。对比命令会按验证集分组，避免把不同验证集的分数直接排在一起。

比较不同训练数据量时，关注原始 `best_auc`；`auc_gain_vs_prior` 是相对各自训练数据得到的点击率基线的增量。

每次优先只改变一个主要因素。AUC 增量为负也是有效结果；保留所有实验记录。小样本中有希望的修改，再使用完整时间划分和另一个切点复核。

## 运行检查

```bash
python3 -m unittest -v test_pipeline.py
```

检查涵盖时间切分、验证标签隔离、冷启动用户、空历史、历史裁剪、固定验证集、模型保存加载、行号匹配及预测文件格式。
