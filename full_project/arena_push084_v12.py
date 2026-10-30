"""Support-weighted exposure MAP in a past-only SGNS8 frame."""
import json
import os
import subprocess
import sys
import time
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v12'
runner.FINAL = ROOT/'competition/final_v12'
runner.NAMES = ['v12_map_loose', 'v12_map_balanced', 'v12_map_strong', 'v12_map_user_only', 'v12_map_full_exposure']
runner.VERIFY_SCRIPT = 'arena_push084_v12.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v12.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'arena_graph_trees.py', 'arena_graph_lgbm.py',
                        'arena_novel_choice.py', 'arena_bayes_mf.py', 'test_arena_bayes_mf.py', 'requirements-graph.txt']
runner.DESCRIPTION = '先固定更早 SGNS8 表示、以其后三天曝光校准 16 个全局系数，再固定尺度进行逐 ID 支持量加权的高斯锚定；比较正则强度、固定物品表示，以及纳入切分前全部真实曝光的经验锚定版本。'


def wait_file(path, completed=False):
    while True:
        if path.exists() and (not completed or json.loads(path.read_text()).get('stage') == 'completed'):
            return
        time.sleep(15)


def preflight():
    runner.OUT.mkdir(parents=True, exist_ok=True)
    wait_file(ROOT/'competition/ood/runs/v11_novel_spectral_cross/2025-01-25/result.json')
    for name in runner.NAMES:
        for day in ['26', '25']:
            p = ROOT/'competition/ood/runs'/name/('2025-01-'+day)/'result.json'
            if p.exists():
                continue
            args = [sys.executable, '-u', 'arena_ood.py', 'train', '--name', name, '--config', 'configs/'+name+'.json', '--cutoff', '2025-01-'+day]
            if day == '25':
                primary = json.loads((p.parent.parent/'2025-01-26/result.json').read_text())
                args += ['--fixed-epochs', str(primary['best_epoch'])]
            print(json.dumps(dict(stage='preflight', name=name, cutoff=day)), flush=True)
            with (runner.OUT/(name+'_'+day+'_preflight.log')).open('a') as log:
                subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    if '--verify' in sys.argv:
        import arena_push084_export as exporter
        exporter.OUT, exporter.FINAL = runner.OUT, runner.FINAL
        exporter.verify()
    else:
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        if '--preflight' in sys.argv:
            preflight()
        else:
            wait_file(ROOT/'competition/push084_v11/state.json', True)
            runner.run()
