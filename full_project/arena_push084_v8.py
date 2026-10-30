"""Sequential follow-up campaign: exact feedback memory and repeat behavior."""
import json
import os
import subprocess
import sys
import time
from arena_ood import ROOT, write_json
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v8'
runner.FINAL = ROOT/'competition/final_v8'
runner.NAMES = ['v8_pair_wide', 'v8_pair_coclick_wide', 'v8_pair_spectral_wide', 'v8_pair_origin_expanding']
runner.VERIFY_SCRIPT = 'arena_push084_v8.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v8.py', 'arena_pair_memory.py', 'test_arena_v7.py', 'test_arena_pair_memory.py']
runner.DESCRIPTION = '显式使用先前用户物品曝光、点击、点击间隔和经验贝叶斯收缩后的重复选择统计；补测共点击、图表示与更长滚动窗口的组合。'


if __name__ == '__main__':
    if '--verify' in sys.argv:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = runner.OUT, runner.FINAL
        exporter.verify()
    else:
        runner.OUT.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        while True:
            state = ROOT/'competition/push084_v7/state.json'
            if state.exists() and json.loads(state.read_text()).get('stage') == 'completed':
                break
            time.sleep(15)
        pointer = json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
        if pointer['validation_26']['auc'] >= .84:
            write_json(runner.OUT/'state.json', dict(stage='target_already_reached', updated=time.time()))
        else:
            runner.run()
