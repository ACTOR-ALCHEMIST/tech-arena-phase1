"""Native history attention; same-frame latent inputs, strict temporal audits."""
import json
import sys
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v9'
runner.FINAL = ROOT/'competition/final_v9'
runner.NAMES = ['v9_din_frozen', 'v9_attention_frozen', 'v9_attention_tuned', 'v9_attention_joint']
runner.VERIFY_SCRIPT = 'arena_push084_v9.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v9.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'test_arena_native_attention.py']
runner.DESCRIPTION = '从已验证的强统计评分器出发，以同一坐标系内的 SGNS32 多视图直接读取逐条历史；对比 DIN、冻结 Transformer、微调 embedding、联合微调评分器。'


if __name__ == '__main__':
    if '--verify' in sys.argv:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = runner.OUT, runner.FINAL
        exporter.verify()
    else:
        assert json.loads((ROOT/'competition/push084_v8/state.json').read_text())['stage'] == 'completed'
        runner.run()
