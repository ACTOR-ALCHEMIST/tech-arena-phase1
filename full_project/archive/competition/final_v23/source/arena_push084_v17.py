"""Fresh-view ensemble diversity, slower fitting, and denser ranking gradients."""
import argparse
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v17'
runner.FINAL = ROOT/'competition/final_v17'
runner.NAMES = ['v17_fresh_seed2027', 'v17_fresh_seed43', 'v17_fresh_slow', 'v17_fresh_rank']
runner.VERIFY_SCRIPT = 'arena_push084_v17.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v17.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'arena_graph_trees.py', 'arena_graph_lgbm.py',
                        'arena_novel_choice.py', 'arena_bayes_mf.py', 'arena_supervised_repr.py',
                        'arena_online_events.py', 'arena_ppmi.py', 'test_arena_rank_sampling.py', 'requirements-graph.txt']
runner.DESCRIPTION = '固定已验证的曝光细调+多阶谱图输入，尝试两个独立种子、较低学习率，以及每个正例256个负例的全局行排序辅助损失。验证流程和候选融合权重网格保持一致。'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.verify:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = runner.OUT, runner.FINAL
        exporter.verify()
    else:
        runner.run()
