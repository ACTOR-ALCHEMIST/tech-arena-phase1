"""Strictly causal, timestamp-grouped exposure-history features."""
import argparse
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v15'
runner.FINAL = ROOT/'competition/final_v15'
runner.NAMES = ['v15_online_wide', 'v15_online_spectral_wide', 'v15_online_fresh_wide', 'v15_online_fresh_cross']
runner.VERIFY_SCRIPT = 'arena_push084_v15.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v15.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'arena_graph_trees.py', 'arena_graph_lgbm.py',
                        'arena_novel_choice.py', 'arena_bayes_mf.py', 'arena_supervised_repr.py',
                        'arena_online_events.py', 'test_arena_online_events.py', 'requirements-graph.txt']
runner.DESCRIPTION = '加入43个严格因果的用户/物品曝光历史特征：支持量、平滑CTR、近期衰减CTR、连续点击/未点击组数、上次曝光间隔和在线先验。同一时间戳聚合，查询前且验证截止前双重截断；比较基础、谱图和曝光细调表示组合。'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.verify:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = runner.OUT, runner.FINAL
        exporter.verify()
    elif args.prepare:
        from arena_online_events import memory
        for cutoff in ['2025-01-26', '2025-01-25']:
            memory(cutoff)
    else:
        runner.run()
