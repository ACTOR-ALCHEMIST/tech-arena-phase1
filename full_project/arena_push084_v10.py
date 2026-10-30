"""A different function family: boosted trees with graph/slate context."""
import json
import os
import subprocess
import sys
import time
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v10'
runner.FINAL = ROOT/'competition/final_v10'
runner.NAMES = ['v10_graph_d6', 'v10_lgbm31', 'v10_lgbm63', 'v10_lgbm63_context']
runner.VERIFY_SCRIPT = 'arena_push084_v10.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v10.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'arena_graph_trees.py', 'arena_graph_lgbm.py',
                        'requirements-graph.txt', 'test_arena_graph_trees.py']
runner.DESCRIPTION = '使用 CPU CatBoost 与 LightGBM 分段评分，融合谱图、历史记忆和候选集合上下文；LightGBM 对比 31/63 个叶子。'


if __name__ == '__main__':
    if '--verify' in sys.argv:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = runner.OUT, runner.FINAL
        exporter.verify()
    else:
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        while True:
            state = ROOT/'competition/push084_v9/state.json'
            if state.exists() and json.loads(state.read_text()).get('stage') == 'completed':
                break
            time.sleep(15)
        runner.run()
