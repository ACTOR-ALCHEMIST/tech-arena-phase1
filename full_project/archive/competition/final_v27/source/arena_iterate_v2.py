"""Resume the second architecture iteration without losing the frozen v1.

python3 arena_iterate_v2.py
python3 arena_iterate_v2.py --select-only
"""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

# A directly runnable entry point on the local 16GB Apple Silicon machine.
for key,value in {'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
                  'PYTORCH_MPS_HIGH_WATERMARK_RATIO':'0.55','PYTORCH_MPS_LOW_WATERMARK_RATIO':'0.4'}.items():
    os.environ.setdefault(key,value)

import numpy as np
from arena_core import ROOT, LAB, evaluate
from arena_finalize import ensure_secondary, prediction_record, blend
from arena_report import note, write_json, now
from arena_validation import importance_weights, weighted_auc

NAMES = ['v2_wide256_s2026', 'v2_cross256_s2026', 'v2_attention128_s2026',
         'v2_multiview_wide256_s2026', 'v2_multiview_cross256_s2026']


def read_result(name):
    return json.loads((LAB/'experiments'/name/'result.json').read_text())


def select():
    incumbent=json.loads((LAB/'champions/v1/manifest.json').read_text())
    sources=incumbent['sources']+NAMES
    records=[read_result(name) for name in sources]
    audits=[ensure_secondary(r) for r in records]
    arrays=[prediction_record(r) for r in records]
    secondary=[prediction_record(r) for r in audits]
    for group in (arrays,secondary):
        for row in group[1:]:
            np.testing.assert_array_equal(row[0],group[0][0])
            np.testing.assert_array_equal(row[1],group[0][1])
    iw26,_=importance_weights('2025-01-26',arrays[0][0])
    iw25,_=importance_weights('2025-01-25',secondary[0][0])
    n=len(sources);old_count=len(incumbent['sources']);weights=[]

    def recipe(old_mass, new_ids):
        w=[old_mass*a for a in incumbent['weights']]+[0.]*(n-old_count)
        for i in new_ids:w[i]=(1-old_mass)/len(new_ids)
        return w

    weights.append(recipe(1,[]))
    for i in range(old_count,n):
        for old_mass in (0.,.25,.5,.75):weights.append(recipe(old_mass,[i]))
    for pair in itertools.combinations(range(old_count,n),2):weights.append(recipe(.5,pair))
    weights.append(recipe(.5,list(range(old_count,n))))
    scores=[]
    for w in weights:
        p=blend([r[2] for r in arrays],w,'logit')
        q=blend([r[2] for r in secondary],w,'logit')
        wa=weighted_auc(arrays[0][1],p,iw26);wb=weighted_auc(secondary[0][1],q,iw25)
        scores.append(dict(kind='logit',weights=w,auc_26=evaluate(arrays[0][1],p)['auc'],
                           auc_25=evaluate(secondary[0][1],q)['auc'],weighted_auc_26=wa,
                           weighted_auc_25=wb,selection_score=(wa+wb)/2))
    baseline=scores[0]
    for day in ('26','25'):
        assert abs(baseline['auc_'+day]-incumbent['validation_'+day]['auc'])<1e-10
    assert abs(baseline['selection_score']-incumbent['selection_score'])<1e-10
    eligible=[s for s in scores[1:] if s['auc_26']>baseline['auc_26']+1e-6
              and s['auc_25']>baseline['auc_25']+1e-6
              and s['selection_score']>baseline['selection_score']+1e-6]
    winner=max(eligible,key=lambda r:r['selection_score']) if eligible else baseline
    report=dict(sources=sources,selected=winner,all_recipes=scores,incumbent=baseline,
                eligible_recipes=len(eligible),primary_results=[dict(name=r['name'],auc=r['auc'],
                best_epoch=r.get('best_epoch'),parameters=r.get('parameters'),seconds=r['seconds']) for r in records],
                secondary_results=[dict(name=r['name'],auc=r['auc']) for r in audits],
                protocol='Same Jan26 first-event/user cohort; primary-selected epochs and scheduler locked on Jan25; predeclared logit mixtures with frozen v1; promotion requires improvement in both raw AUCs and mean cohort-weighted AUC. Jan24 audit is not used to choose weights.')
    write_json(LAB/'v2_selection.json',report)
    note(f"v2 固定粗网格比较完成：主切点 {winner['auc_26']:.6f}，第二切点 {winner['auc_25']:.6f}，满足双切点改进的配方 {len(eligible)} 个。",'v2')
    return report


def promote():
    final=LAB/'final_v2'
    manifest=json.loads((final/'manifest.json').read_text())
    acceptance=json.loads((final/'acceptance.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert acceptance['passed'] and acceptance['manifest_sha256']==digest(final/'manifest.json')
    assert digest(final/'submission.csv')==manifest['sha256']
    incumbent=json.loads((LAB/'champions/v1/manifest.json').read_text())
    assert digest(LAB/'champions/v1/submission.csv')==incumbent['sha256']
    current=json.loads((LAB/'final/manifest.json').read_text())
    assert current['sha256'] in (incumbent['sha256'],manifest['sha256']),'Another iteration changed the incumbent'
    for day in ('26','25'):
        assert manifest['validation_'+day]['auc']>incumbent['validation_'+day]['auc']+1e-6
    assert manifest['selection_score']>incumbent['selection_score']+1e-6
    for name in ('submission.csv','acceptance.json','manifest.json'):
        partial=LAB/'final'/(name+'.v2.partial')
        shutil.copy2(final/name,partial);os.replace(partial,LAB/'final'/name)
    assert digest(LAB/'final/submission.csv')==manifest['sha256']


def write_comparison():
    """Paired event bootstrap against the retained incumbent, after selection."""
    from arena_verify import clustered_bootstrap
    old=json.loads((LAB/'champions/v1/manifest.json').read_text())
    new=json.loads((LAB/'final_v2/manifest.json').read_text())
    comparisons={}
    for day in ('26','25','24'):
        a=np.load(ROOT/old['model_dir']/f'validation_{day}.npz')
        b=np.load(ROOT/new['model_dir']/f'validation_{day}.npz')
        np.testing.assert_array_equal(a['event_indices'],b['event_indices'])
        np.testing.assert_array_equal(a['labels'],b['labels'])
        interval=clustered_bootstrap(b['labels'],b['predictions'],a['predictions'])
        comparisons['2025-01-'+day]={
            'v1_auc':old['validation_'+day]['auc'],'v2_auc':new['validation_'+day]['auc'],
            'gain':new['validation_'+day]['auc']-old['validation_'+day]['auc'],
            'gain_vs_v1_95_ci':interval['gain_vs_official_starter_95_ci'],
            'repeats':interval['repeats'],'method':interval['method']}
    write_json(LAB/'v2_comparison.json',dict(v1_sha256=old['sha256'],v2_sha256=new['sha256'],
        cutoffs=comparisons,official_evaluation_labels_available=False,
        limitation='Intervals condition on the selected recipe and do not account for repeated model and blend selection.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--select-only',action='store_true');args=parser.parse_args()
    with (LAB/'v2.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.select_only:
            from arena_tabular import run
            for name in NAMES:
                folder=LAB/'experiments'/name
                if not (folder/'result.json').exists():run(json.loads((folder/'config.json').read_text()))
        report=select()
        if args.select_only:return
        if report['eligible_recipes']:
            incumbent=json.loads((LAB/'champions/v1/manifest.json').read_text())
            commands=[['arena_finalize.py','--final-dir','competition/final_v2','--selection-file','competition/v2_selection.json',
                       '--reuse-model-dir',incumbent['model_dir']],['arena_verify.py','--final-dir','competition/final_v2']]
            # The export directory is resumable once its verified manifest exists.
            for command in commands:
                if command[0]=='arena_finalize.py' and (LAB/'final_v2/manifest.json').exists():continue
                subprocess.run([sys.executable,'-u']+command,cwd=ROOT,check=True)
            promote()
            write_comparison()
        state=json.loads((LAB/'iteration_v2.json').read_text())
        state.update(status='completed; v2 promoted' if report['eligible_recipes'] else 'completed; frozen v1 retained',
                     finished_at=now(),report='competition/v2_selection.json')
        write_json(LAB/'iteration_v2.json',state)
        note('v2 结构扩展已完成并独立验收，新版晋升；v1 完整保留。' if report['eligible_recipes'] else
             'v2 结构扩展完成，未满足双切点一致改进条件，继续推荐冻结的 v1；新结构配置、模型与指标全部保留。','v2')


if __name__=='__main__':main()
