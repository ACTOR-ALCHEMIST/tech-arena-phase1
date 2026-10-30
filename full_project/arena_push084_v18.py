"""Past-only spherical preference vectors with fixed two-view descriptors."""
import argparse
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v18'
runner.FINAL = ROOT/'competition/final_v18'
runner.NAMES = ['v18_sphere_wide', 'v18_sphere_spectral_wide', 'v18_sphere_fresh_wide', 'v18_sphere_fresh_cross']
runner.VERIFY_SCRIPT = 'arena_push084_v18.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v18.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'arena_graph_trees.py', 'arena_graph_lgbm.py',
                        'arena_novel_choice.py', 'arena_bayes_mf.py', 'arena_supervised_repr.py',
                        'arena_online_events.py', 'arena_ppmi.py', 'arena_spherical_probe.py', 'arena_spherical_repr.py',
                        'test_arena_spherical.py', 'requirements-graph.txt']
runner.DESCRIPTION = '固定8维单位球偏好向量，以历史二值点击继续训练共享温度和偏好方向；固定第16/32轮的44个描述特征，比较基础、谱图、曝光细调的组合与宽/交叉网络。表示训练不读取验证目标。'


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
        from arena_spherical_repr import model
        for cutoff in ['2025-01-26', '2025-01-25', '2025-01-24', '2025-01-23']:
            model(cutoff)
    else:
        runner.run()
