# 继续向 0.84：push084_v9

不可变基线：competition/final_v8/submission.csv。从已验证的强统计评分器出发，以同一坐标系内的 SGNS32 多视图直接读取逐条历史；对比 DIN、冻结 Transformer、微调 embedding、联合微调评分器。所有拟合目标在各自切分之前。

| 配置 | Jan26 | Jan25 | 轮数 |
|---|---:|---:|---:|
| v9_din_frozen | 0.830737956 | 0.820242087 | 3 |
| v9_attention_frozen | 0.830729082 | 0.820162067 | 4 |
| v9_attention_tuned | 0.830757744 | 0.819823211 | 5 |
| v9_attention_joint | 0.830696588 | 0.819978239 | 1 |

晋升新版本：True。当前最优 CSV：[competition/final_v9/submission.csv](../../competition/final_v9/submission.csv)。

本地主切分 0.831399146；0.84 尚未达到。官方标签不可用。

Jan26：0.831382749 → 0.831399146（+0.000016397）。
Jan25：0.821414586 → 0.821470227（+0.000055642）。
Jan24：0.807796971 → 0.808018545（+0.000221574）。