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
