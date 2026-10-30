"""Evidence-driven removal/dropout of fragile temporal feature groups."""
import json
import os
import subprocess
import sys
import time
from arena_ood import ROOT
import arena_push084_v7 as runner

runner.OUT = ROOT/'competition/push084_v13'
runner.FINAL = ROOT/'competition/final_v13'
runner.NAMES = ['v13_drop_semantic', 'v13_drop_affinity', 'v13_drop_all_temporal', 'v13_temporal_dropout']
runner.VERIFY_SCRIPT = 'arena_push084_v13.py'
runner.EXTRA_SOURCES = ['arena_residual_adapter.py', 'arena_push084_v7.py', 'arena_push084_v13.py',
                        'arena_pair_memory.py', 'arena_native_attention.py', 'arena_graph_trees.py', 'arena_graph_lgbm.py',
                        'arena_novel_choice.py', 'arena_bayes_mf.py', 'test_arena_feature_mask.py', 'requirements-graph.txt']
runner.DESCRIPTION = '由冻结评分器的 Jan25/26 时间特征替换诊断提出：固定移除时段语义匹配、时段偏好统计、全部这两类特征，或训练时按完整事件随机屏蔽；保留原始历史与候选、标签。'


def wait_for(path, complete=False):
    while True:
        if path.exists() and (not complete or json.loads(path.read_text()).get('stage') == 'completed'):
            return
        time.sleep(15)


def preflight():
    runner.OUT.mkdir(parents=True, exist_ok=True)
    wait_for(ROOT/'competition/ood/runs/v12_map_full_exposure/2025-01-25/result.json')
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
            wait_for(ROOT/'competition/push084_v12/state.json', True)
            runner.run()
