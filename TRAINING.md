# 完整训练与研发源码

[`full_project/`](full_project/) 包含原项目的全部自有源码：**1,812 份代码文件，其中 185 份为项目根目录的当前代码，其余包含历史源码快照**；另外保留配置、依赖清单、测试及文本实验记录。原项目的第三方安装目录、虚拟环境及生成的二进制产物没有纳入本次源码提交。

根目录原有的 `predict.py` 和 `assets/` 继续用于复现已验收 V31 的提交结果。

## 目录

| 路径 | 内容 |
|---|---|
| `full_project/*.py` | 当前的数据处理、训练、验证、推理、导出和测试代码 |
| `full_project/overnight*.go` | Go 实验调度器和测试 |
| `full_project/configs/` | 103 份实验配置，包括 V31、Transformer、生成式 LoRA 等配置 |
| `full_project/requirements*.txt` | 各条训练路线的原始依赖版本 |
| `full_project/README.md`、`LIGHT_START.md` | 原项目方案记录和轻量训练说明 |
| `full_project/archive/competition/` | 历史版本源码快照、配置、训练记录、模型选择和验收记录 |
| `full_project/archive/runs/`、`archive/model/` | 早期轻量实验源码快照及文本记录 |
| `full_project/SOURCE_INVENTORY.json` | 全部 5,106 份复制文件的原路径、归档路径、大小和 SHA-256 |
| `full_project/ORIGINAL_GITIGNORE.txt` | 原工作区的忽略规则备份 |

当前代码及 `configs/` 保持原始相对位置。历史目录整体移至 `archive/`，目录内部保留原结构，避免缺少二进制缓存时误用旧的“训练完成”标记。

## 先运行基础训练链路

在仓库根目录执行，使用 Python 3.12：

```bash
cd full_project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-arena.txt
```

提供比赛原始 `train.csv` 的实际路径，生成事件级训练缓存。evaluation 使用仓库现有输入：

```bash
python run.py prepare --train /absolute/path/to/train.csv --evaluation ../assets/evaluation.zip
```

运行轻量双塔训练，观察常数、点击率基线和训练后指标：

```bash
python light.py --name fresh_smoke --epochs 2 --skip-predict
```

这个名称每次运行需保持唯一。`--skip-predict` 只跳过最后的整份 evaluation 预测，训练与本地验证仍会执行。

使用完整训练窗口训练和重训基础双塔：

```bash
python run.py train --device cpu
python run.py retrain --device cpu
python run.py predict --device cpu --evaluation ../assets/evaluation.zip --out submissions/two_tower.csv
```

这些命令训练基础双塔。V31 最终集成使用下面列出的表示、神经网络和多阶段修正代码。

## 主要训练入口

| 训练部分 | 当前源码 |
|---|---|
| CSV 解析、时间划分、ID 编码 | `ctr_data.py`、`run.py`、`light_data.py` |
| 双塔基线 | `ctr_model.py`、`light.py`、`run.py` |
| 共现、ALS、SGNS、谱图表示 | `arena_core.py`、`arena_latent.py`、`arena_sgns.py`、`arena_spectral.py` |
| 曝光监督 MF 与新鲜表示 | `arena_bayes_mf.py`、`arena_supervised_repr.py`、`arena_robust_mf.py` |
| 球面表示、多种子融合 | `arena_spherical*.py`、`arena_fused_sphere.py`、`arena_unit_consensus.py` |
| 数值特征、MLP、事件上下文 | `arena_features.py`、`arena_ood_features.py`、`arena_tabular.py`、`arena_neural_v2.py`、`arena_push_model.py` |
| Transformer、原生历史注意力 | `arena_v3*.py`、`arena_native_attention.py`、`arena_history_*.py` |
| LLM、生成式分类、LoRA | `generative*.py`、`setup_generative.py`、`configs/generative*.json` |
| OOD、历史删除与增强 | `arena_ood.py`、`arena_ood_iterate.py`、`arena_ood_audit.py` |
| 精排、重排、条件偏好修正 | `arena_two_stage*.py`、`arena_metric_*.py`、`arena_conditional*.py`、`arena_latent_delta*.py` |
| V31 实验控制器 | `arena_push084_v31.py`、`configs/v31_unit_*wide.json` |
| 全量重训、融合与导出 | `arena_ood_export.py`、`arena_push084_export.py`、`arena_finalize.py` |
| 回归测试 | `test_*.py`、`overnight_test.go` |

生成式 MLX 路线使用单独环境及 `requirements-generative.txt`，依赖 Apple Silicon；Torch Transformer 路线使用 `requirements-v3.txt`，图树路线还需 `requirements-graph.txt`。历史控制器中含 macOS 的 `caffeinate` 和 Unix 文件锁调用，跨平台运行需要相应适配。

## 高级实验与复现范围

原始 `train.csv` 约 7.3 GB，历史模型、特征和实验二进制产物合计数十 GB，未上传到本次源码提交。`archive/` 中的 JSON、验收记录和模型路径是历史记录，对应的权重、验证 NPZ、特征 NPY 和历史提交 CSV 需要按训练流程重新生成或从原工作区恢复。

高级实验控制器会检查历史基线、缓存及哈希，尤其 V31 依赖早期特征方案和已验收集成。完整源码已经提供；从空目录开始直接运行一次 `arena_push084_v31.py` 并不能替代完整的前置训练流程。历史源码和配置按原样保存，没有为这次上传重写为新的全自动训练管线。

需要立即得到与已验收 V31 逐字节相同的提交 CSV，使用仓库根目录 [README](README.md) 中的推理流程。跨设备从头训练的数值结果不承诺逐位一致。

## 本次源码核对

所有复制文件均与原工作区逐文件核对 SHA-256；1,806 份 Python 文件通过语法解析。源码没有因上传而修改。本次未重新进行模型训练，也没有把旧验收记录描述成一次新的训练验收。
