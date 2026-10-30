"""A different representation objective: non-repeat real choice likelihood."""
import json
import os
import subprocess
import sys
import time
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v11'
runner.FINAL = ROOT/'competition/final_v11'
runner.NAMES = ['v11_novel_wide', 'v11_novel_cross', 'v11_novel_spectral_wide', 'v11_novel_spectral_cross']
runner.VERIFY_SCRIPT = 'arena_push084_v11.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v11.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'arena_graph_trees.py', 'arena_novel_choice.py',
                        'test_arena_novel_choice.py']
runner.DESCRIPTION = '真实曝光偏好表示先固定 embedding 拟合尺度，再以强锚定小学习率更新；偏好目标只使用非重复点击事件，并排除重复候选；对比第 12/40 轮表示及图特征组合。'


def wait_for(name):
    while True:
        p = ROOT/('competition/'+name+'/state.json')
        if p.exists() and json.loads(p.read_text()).get('stage') == 'completed':
            return
        time.sleep(15)


if __name__ == '__main__':
    if '--verify' in sys.argv:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = runner.OUT, runner.FINAL
        exporter.verify()
    elif '--prepare' in sys.argv:
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        wait_for('push084_v9')
        from arena_novel_choice import model
        for cutoff in ['2025-01-24', '2025-01-26', '2025-01-23', '2025-01-25']:
            model(cutoff)
    else:
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        wait_for('push084_v10')
        runner.run()
