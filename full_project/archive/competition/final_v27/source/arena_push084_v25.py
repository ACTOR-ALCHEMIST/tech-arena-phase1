"""Independent-seed preference fusion, after the long-only campaign."""
import argparse
from arena_ood import ROOT
import arena_push084_v24 as campaign

runner=campaign.runner
runner.OUT=ROOT/'competition/push084_v25';runner.FINAL=ROOT/'competition/final_v25'
runner.NAMES=['v25_fused_wide','v25_fused_spectral_wide','v25_fused_fresh_wide','v25_fused_stack_wide']
runner.VERIFY_SCRIPT='arena_push084_v25.py'
runner.EXTRA_SOURCES=runner.EXTRA_SOURCES+['arena_push084_v25.py','arena_fused_sphere.py','arena_aligned_repr.py','test_arena_fused_sphere.py']
runner.DESCRIPTION='把固定128轮单位球表示与独立seed42的SGNS8做过去物品表上的加权正交对齐，保留50%/75%混合向量的方向与一致性范数；用相同宽网络比较基础、谱图、曝光细调和全部旧/新表示，最大输入380维。融合本身不读取查询标签。'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:
        import arena_push084_export as exporter
        exporter.OUT,exporter.FINAL=runner.OUT,runner.FINAL;exporter.verify()
    else:campaign.run()
