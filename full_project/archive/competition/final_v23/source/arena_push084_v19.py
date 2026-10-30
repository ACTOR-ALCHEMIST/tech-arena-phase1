"""Nonlinear scoring of fresh latent coordinates in a fixed historical frame."""
import argparse
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v19'
runner.FINAL = ROOT/'competition/final_v19'
runner.NAMES = ['v19_aligned_wide', 'v19_aligned_spectral_wide', 'v19_aligned_sphere_wide', 'v19_aligned_fresh_cross']
runner.VERIFY_SCRIPT = 'arena_push084_v19.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v19.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'arena_graph_trees.py', 'arena_graph_lgbm.py',
                        'arena_novel_choice.py', 'arena_bayes_mf.py', 'arena_supervised_repr.py',
                        'arena_online_events.py', 'arena_ppmi.py', 'arena_spherical_probe.py', 'arena_spherical_repr.py',
                        'arena_aligned_repr.py', 'test_arena_aligned_repr.py', 'requirements-graph.txt']
runner.DESCRIPTION = '把每个历史截点的单位球向量正交对齐到固定Jan21参考，增加用户/物品向量分量、历史偏好分量和逐维交互，共75列；比较基础、谱图、球面描述和曝光细调组合。对齐与表示均不读取查询标签，最终快照可独立重放。'


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
