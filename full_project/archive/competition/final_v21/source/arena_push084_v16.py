"""A separate shrunk-PMI graph view with locked temporal validation."""
import argparse
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v16'
runner.FINAL = ROOT/'competition/final_v16'
runner.NAMES = ['v16_ppmi_wide', 'v16_ppmi_spectral_wide', 'v16_ppmi_fresh_wide', 'v16_ppmi_fresh_cross']
runner.VERIFY_SCRIPT = 'arena_push084_v16.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v16.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'arena_graph_trees.py', 'arena_graph_lgbm.py',
                        'arena_novel_choice.py', 'arena_bayes_mf.py', 'arena_supervised_repr.py',
                        'arena_online_events.py', 'arena_ppmi.py', 'test_arena_ppmi.py', 'requirements-graph.txt']
runner.DESCRIPTION = '新增按物品流行度校正、共现次数及节点强度收缩的正PMI物品图，提取16/32维图表示的44个历史匹配特征，以及9个修复稀疏去对角问题后的精确共现特征；比较单独视图、标准谱图组合和曝光细调组合。全部表示严格早于对应训练/验证窗口。'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = runner.OUT, runner.FINAL
        exporter.verify()
    else:
        runner.run()
