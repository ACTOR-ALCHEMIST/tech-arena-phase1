"""Expose the accepted unit consensus directly to strong numeric rankers."""
import argparse
import json
import os
import subprocess
import sys
import time
from arena_ood import ROOT,write_json,sha,guard
import arena_push084_v7 as runner

runner.OUT=ROOT/os.environ.get('ARENA_UNIT_OUT','competition/push084_v31')
runner.FINAL=ROOT/os.environ.get('ARENA_UNIT_FINAL','competition/final_v31')
runner.NAMES=['v31_unit_fresh_wide','v31_unit_long_fresh_wide']
runner.VERIFY_SCRIPT='arena_push084_v31.py';runner.TARGET_AUC=.838
runner.EXTRA_SOURCES=[p.name for p in ROOT.glob('arena_*.py')]+['test_arena_unit_consensus.py']
runner.DESCRIPTION='将已在V26通过时间验证的两组单位球向量共识直接暴露给宽精排网络，加入31个偏好/历史/新旧表示差值特征；比较曝光细调+谱图底座及额外保留64/128轮长表示两个配置。区别于V25的原始SGNS混合。'


def cleanup(days):
    path=runner.OUT/'working_cache_cleanup.json'
    ledger=json.loads(path.read_text()) if path.exists() else []
    for day in days:
        for folder in (ROOT/'competition/ood/features').glob('2025-01-'+day+'_*'):
            if not any(folder.name.endswith('_'+key) for key in ['fresh_mf','sphere_long','unit_consensus']):continue
            f=folder/'train.npy'
            if f.exists():
                assert (folder/'train.json').exists()
                ledger.append(dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f),
                                   reason='Regenerable feature matrix; all consuming rankers finished. Models, labels, indices and validation matrices preserved.'))
                write_json(path,ledger);f.unlink()


def finish():
    # Full ranker fitting has completed under --train-only before this step.
    selection=json.loads((runner.OUT/'selection.json').read_text())
    assert json.loads((runner.OUT/'incumbent_pointer.json').read_text())==json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    cleanup(['26','25']);before=guard();reports=runner.audit();cleanup(['24'])
    stable=selection['eligible']>0
    if stable:
        third=reports['24'];anchor=json.loads((ROOT/'competition/final_v6/audit_24.json').read_text())
        stable=(third['delta']>=-.0001 and third['cohorts']['weekday']['delta']>=-.0001 and
                third['new_auc']>=anchor['new_auc']-.0001 and third['cohorts']['weekday']['new_auc']>=anchor['cohorts']['weekday']['new_auc']-.0001)
    write_json(runner.OUT/'promotion.json',dict(promoted=bool(stable),target_auc=.838))
    assert guard()==before
    if stable:
        import arena_push084_export as exporter
        exporter.OUT,exporter.FINAL,exporter.TARGET_AUC=runner.OUT,runner.FINAL,.838
        exporter.export(verify_script=runner.VERIFY_SCRIPT,extra_sources=runner.EXTRA_SOURCES)
        code="import sys;from pathlib import Path;sys.path.insert(0,str(Path(sys.argv[1])/'source'));import arena_push084_export as e;e.ROOT=Path.cwd();e.FINAL=Path(sys.argv[1]);e.verify()"
        subprocess.run([sys.executable,'-u','-c',code,str(runner.FINAL)],check=True,cwd=ROOT)
    runner.report(reports,bool(stable))
    write_json(runner.OUT/'state.json',dict(stage='completed',promoted=bool(stable),updated=time.time(),target_auc=.838))
    print(json.dumps(dict(stage='unit_consensus_completed',promoted=bool(stable))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--train-only',action='store_true');p.add_argument('--finish',action='store_true');a=p.parse_args()
    if a.verify:
        import arena_push084_export as exporter
        exporter.OUT,exporter.FINAL=runner.OUT,runner.FINAL;exporter.verify()
    elif a.finish:finish()
    else:
        runner.OUT.mkdir(parents=True,exist_ok=True)
        runner.campaign_module().train()
        import arena_push084_reblend as refine
        refine.OUT=runner.OUT;refine.run()
        if not a.train_only:finish()
