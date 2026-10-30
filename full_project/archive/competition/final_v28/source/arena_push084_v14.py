"""Strong numeric heads over fresh exposure-refined, past-only embeddings."""
import argparse
import json
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v14'
runner.FINAL = ROOT/'competition/final_v14'
runner.NAMES = ['v14_fresh_wide', 'v14_fresh_cross', 'v14_fresh_spectral_wide', 'v14_fresh_spectral_cross']
runner.VERIFY_SCRIPT = 'arena_push084_v14.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v14.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'arena_graph_trees.py', 'arena_graph_lgbm.py',
                        'arena_novel_choice.py', 'arena_bayes_mf.py', 'arena_supervised_repr.py',
                        'test_arena_supervised_repr.py', 'test_arena_bayes_mf.py', 'requirements-graph.txt']
runner.DESCRIPTION = '在每个切分的完整历史 SGNS8 上加入真实曝光负反馈；评分尺度先在更早的表示上校准并固定，再按历史支持量正则化细调。固定第4/16轮的48个辅助特征，比较宽 MLP/交叉网络和多阶谱图组合。'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    if args.verify:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = runner.OUT, runner.FINAL
        exporter.verify()
    elif args.prepare:
        from arena_supervised_repr import model
        for cutoff in ['2025-01-24', '2025-01-26', '2025-01-23', '2025-01-25']:
            model(cutoff)
    else:
        assert json.loads((ROOT/'competition/push084_v13/state.json').read_text())['stage'] == 'completed'
        runner.run()
