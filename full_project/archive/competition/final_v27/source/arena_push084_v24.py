"""Longer spherical preferences: fixed64/128 views in four numeric heads."""
import argparse
import json
import os
import subprocess
import sys
import time
from arena_ood import ROOT,guard,write_json
import arena_push084_v7 as runner

runner.OUT=ROOT/'competition/push084_v24'
runner.FINAL=ROOT/'competition/final_v24'
runner.NAMES=['v24_long_wide','v24_long_spectral_wide','v24_long_fresh_wide','v24_long_stack_cross']
runner.VERIFY_SCRIPT='arena_push084_v24.py'
runner.TARGET_AUC=.838
runner.EXTRA_SOURCES=['arena_push084_v7.py','arena_push084_v24.py','arena_residual_adapter.py',
    'arena_pair_memory.py','arena_native_attention.py','arena_graph_trees.py','arena_graph_lgbm.py',
    'arena_novel_choice.py','arena_bayes_mf.py','arena_supervised_repr.py','arena_online_events.py',
    'arena_ppmi.py','arena_spherical_probe.py','arena_spherical_repr.py','arena_spherical_long.py',
    'test_arena_spherical_long.py','requirements-graph.txt']
runner.DESCRIPTION='单位球偏好向量按固定128轮学习率计划训练；预先固定第64/128轮的44个描述，测试基础/谱图/曝光细调，以及保留原16/32轮向量的叠加交叉模型。所有表示严格早于查询目标。'


def prebuild(frames):
    processes=[]
    for frame in frames:
        log=(runner.OUT/('representation_'+frame+'.log')).open('a')
        args=[sys.executable,'-u','-c',f'from arena_spherical_long import model; model("{frame}")']
        p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        processes.append((p,log))
    write_json(runner.OUT/'representation_processes.json',dict(frames=frames,pids=[p.pid for p,_ in processes],updated=time.time()))
    for p,log in processes:
        code=p.wait();log.close()
        if code:raise subprocess.CalledProcessError(code,p.args)


def run():
    runner.OUT.mkdir(parents=True,exist_ok=True)
    subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    campaign=runner.campaign_module();campaign.train()
    import arena_push084_reblend as refine
    refine.OUT=runner.OUT;refine.run()
    selection=json.loads((runner.OUT/'selection.json').read_text())
    if selection['eligible']:
        # Only the accepted-looking recipe needs the earlier audit and final
        # representation. Independent CPU fits run together before audit/export.
        prebuild(['2025-01-22','2025-01-27'])
    before=guard();reports=runner.audit()
    selection=json.loads((runner.OUT/'selection.json').read_text())
    stable=selection['eligible']>0 and reports['24']['delta']>=-.0001 and reports['24']['cohorts']['weekday']['delta']>=-.0001
    anchor=None
    if '24' in reports:
        a=json.loads((ROOT/'competition/final_v6/audit_24.json').read_text())
        anchor=dict(overall_floor=a['new_auc']-.0001,weekday_floor=a['cohorts']['weekday']['new_auc']-.0001)
        anchor['passed']=reports['24']['new_auc']>=anchor['overall_floor'] and reports['24']['cohorts']['weekday']['new_auc']>=anchor['weekday_floor']
        stable=stable and anchor['passed']
    write_json(runner.OUT/'promotion.json',dict(promoted=bool(stable),absolute_ood_anchor=anchor,third_origin_tolerance=.0001))
    assert guard()==before
    if stable:
        import arena_push084_export as exporter
        exporter.OUT,exporter.FINAL,exporter.TARGET_AUC=runner.OUT,runner.FINAL,.838
        exporter.export(verify_script=runner.VERIFY_SCRIPT,extra_sources=runner.EXTRA_SOURCES)
    runner.report(reports,bool(stable))
    write_json(runner.OUT/'state.json',dict(stage='completed',promoted=bool(stable),updated=time.time(),target_auc=.838))
    print(json.dumps(dict(stage='completed',promoted=bool(stable),selected=selection['selected'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:
        import arena_push084_export as exporter
        exporter.OUT,exporter.FINAL=runner.OUT,runner.FINAL;exporter.verify()
    else:run()
