# V31 最小独立复现包

**完整训练与研发源码已补充在 [`full_project/`](full_project/)，训练入口见 [TRAINING.md](TRAINING.md)。** 包含全部项目源码、测试、配置，以及历史源码快照和文本实验记录。下文介绍仓库根目录的 V31 推理包。

复制整个文件夹即可运行，不需要原项目、训练缓存、GPU、API Key 或运行时联网。包内包含本次比赛的 `evaluation.zip`，资产约 202 MB，推理代码只有三个文件。

这是**已有权重的 V31 增量推理包**：重新计算 V31 神经网络，再与包内冻结的 V30 预测融合。它不包含 V1–V30 的训练及重新推理流程，也不支持更换为另一批 evaluation 数据。

## 运行

使用 Python **3.12**，在本目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python predict.py
```

Windows 激活环境使用 `.venv\Scripts\activate`；其余命令相同。依赖安装需要联网，之后可离线运行。

输出本目录的 `submission.csv` 和 `submission.csv.verification.json`。默认路径相对于脚本所在目录，因此也可以从其他工作目录运行。**提交 `submission.csv` 即可**。

已在 macOS / Apple M2 / Python 3.12.2 的全新虚拟环境中验证：首次 CPU 推理约 54 秒，峰值常驻内存约 2.1 GB；其他平台的耗时和依赖安装尚未验证。

标准 CSV 共 **529,390** 行，SHA-256 必须是：

```text
b9de517f489bb800a657702acaf8d9fdec41e4b1f62ad545730509164104bbbb
```

## “结果相同”的保证范围

默认命令会检查所有资产的 SHA-256，解析原始 evaluation，重新生成 323 维特征、执行 CPU 神经网络推理、融合 V30 分数，并与冻结结果核对。数值检查通过后，才解压标准 CSV，检查行数、唯一 rowid、概率范围与文件 SHA，再原子写入输出。

**逐字节一致来自经过校验的标准 CSV。** CPU、系统及数学库之间的浮点运算不能保证逐位一致，因此没有把这种标准化导出说成“任意设备重新计算都完全相同”。两个数值检查的最大绝对误差上限均为 `2e-5`；失败会报错，不输出标准结果。已验证机器上的实际误差见 [VERIFICATION.json](VERIFICATION.json)。

需要直接查看本机新计算的概率，可执行：

```bash
python predict.py --numerical-output --out numerical.csv
```

此模式仍做数值核对，但写入新计算值，**不保证 CSV 哈希一致**。不要用它替代已验收提交。

如果只需要导出已验收的 CSV：

```bash
python3 predict.py --replay
```

`--replay` 只需要 Python 3.11+ 标准库，不需要安装依赖，**不执行模型推理**。它仍检查资产完整性及输出 SHA。

## 固定方案

- 特征：基础协同过滤、ALS、时间上下文和 SGNS 共 112 维；谱图 88 维；曝光监督 MF 48 维；长训练球面表示 44 维；双表示共识 31 维，合计 323 维。
- 网络：候选共享 MLP `512 → 256 → 128`，64 维事件上下文分支，10 个候选与 1 个“不点击”类别联合 softmax。实际选中的 V31 使用 MLP；历史 Transformer 尝试未进入这个最终分支。
- 最终融合：`sigmoid(0.95 × logit(V30) + 0.05 × logit(V31网络))`，logit 前裁剪到 `[1e-7, 1-1e-7]`。V30 作为冻结的上游预测输入保存在 `ensemble.npz`。
- 神经网络最终固定训练 7 轮、293,993 个 Jan24–Jan26 事件；训练特征来自 Jan24 前的快照，推理表示来自 Jan27 前的快照。包内只保留推理权重。

| 本地时间切分 | ROC AUC |
|---|---:|
| Jan26 | 0.832004960 |
| Jan25 | 0.822609304 |
| Jan24 | 0.808895070 |

这是本地验证结果，官方 evaluation 没有标签，不能据此声称公开榜分数或已达到 0.838。

## 文件

| 文件 | 用途 |
|---|---|
| `predict.py` | 唯一入口：输入解析、推理、融合、核对、导出 |
| `features.py` | 从冻结版本提取的必要数值特征算法 |
| `models.py` | 曝光 MF 与 V31 MLP 的纯推理网络 |
| `requirements.txt` | 经验证的依赖版本 |
| `manifest.json` | 特征顺序、权重、源资产记录、校验哈希 |
| `assets/evaluation.zip` | 原始比赛预测输入 |
| `assets/vocabulary.json.gz` | 原始 opaque ID 到固定索引的映射 |
| `assets/co_counts.npz`、`representations.npz` | 过去数据拟合的稀疏共现统计、先验和向量表 |
| `assets/weights.pt` | 必要 MF 权重及最终 MLP 权重 |
| `assets/ensemble.npz` | 冻结 V30 概率及 V31 网络数值核对基准 |
| `assets/reference.csv.gz` | 用于逐字节一致导出的已验收 V31 标准结果 |
| `VERIFICATION.json` | 独立目录和全新环境的验证记录 |

只分发这个文件夹即可；`.venv/`、`__pycache__/` 和自行生成的诊断输出无需分发。
