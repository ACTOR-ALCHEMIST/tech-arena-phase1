"""Fail-closed acceptance checks for the delivered model and submission."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from arena_core import ROOT, LAB, evaluate
from arena_features import complete_features
from arena_finalize import explicit_feature_recipe, prediction_record,blend
from arena_report import note, now
from ctr_data import load_cache, temporal_split
from run import check_submission, timestamp


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def check_experiment(name, data):
    folder=LAB/'experiments'/name
    result=explicit_feature_recipe(json.loads((folder/'result.json').read_text()))
    indices,labels,predictions=prediction_record(result)
    _,expected=temporal_split(data,timestamp(result['cutoff']))
    np.testing.assert_array_equal(indices,expected)
    np.testing.assert_array_equal(labels.reshape(-1,10),data['labels'][indices])
    assert abs(roc_auc_score(labels.ravel(),predictions.ravel())-result['auc'])<1e-10
    checks={'name':name,'auc':result['auc'],'cutoff':result['cutoff'],'events':len(indices)}
    if 'feature_dataset' in result:
        features_dir=ROOT/result['feature_dataset'];saved=np.load(features_dir/'labels.npz')
        config=result.get('feature_config',result['config'])
        sample=np.linspace(0,len(indices)-1,min(200,len(indices)),dtype=int)
        actual,names=complete_features(data,indices[sample],result['cutoff'],config)
        stored=np.load(features_dir/'valid.npy',mmap_mode='r')
        rows=(sample[:,None]*10+np.arange(10)).ravel()
        np.testing.assert_allclose(actual,stored[rows],rtol=1e-6,atol=1e-6)
        checks['validation_feature_max_error']=float(np.max(np.abs(actual-stored[rows])))
        metadata=json.loads((features_dir/'metadata.json').read_text())
        assert names==metadata['feature_names']
        ti=saved['train_indices'];sample=np.linspace(0,len(ti)-1,min(100,len(ti)),dtype=int)
        assert (data['timestamps'][ti]<timestamp(result['cutoff'])).all()
        if not config.get('rolling'):
            actual,_=complete_features(data,ti[sample],config.get('train_start','2025-01-22'),config)
            stored_train=np.load(features_dir/'train.npy',mmap_mode='r')
            rows=(sample[:,None]*10+np.arange(10)).ravel()
            np.testing.assert_allclose(actual,stored_train[rows],rtol=1e-6,atol=1e-6)
            checks['training_feature_max_error']=float(np.max(np.abs(actual-stored_train[rows])))
        # Verify the native saved estimator, not only the stored metric.
        if result['method']=='tabular_mlp_mps':
            import torch
            from arena_tabular import build_model,predict
            torch.set_num_threads(2)
            saved_model=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            state=saved_model['state_dict']
            model=build_model(saved_model['width'],state['mean'].numpy(),state['scale'].numpy(),saved_model['config']).eval()
            model.load_state_dict(state);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model.to(device)
            reloaded=predict(model,stored,device)
            np.testing.assert_allclose(reloaded,predictions,rtol=1e-5,atol=2e-6)
            checks['native_model_prediction_max_error']=float(np.max(np.abs(reloaded-predictions)))
        elif result['method'].startswith('catboost'):
            from catboost import CatBoostClassifier
            model=CatBoostClassifier();model.load_model(str(folder/'model.cbm'))
            reloaded=model.predict_proba(stored,thread_count=6)[:,1].reshape(-1,10)
            if result['method']=='catboost_event':
                from arena_event import aggregate,event_probability
                gate=CatBoostClassifier();gate.load_model(str(folder/'event_gate.cbm'))
                reloaded=event_probability(reloaded,gate.predict_proba(aggregate(stored))[:,1],result['temperature'],result['event_mix'])
            np.testing.assert_allclose(reloaded,predictions,rtol=1e-6,atol=1e-7)
            checks['native_model_prediction_max_error']=float(np.max(np.abs(reloaded-predictions)))
    return checks


def clustered_bootstrap(labels,prediction,baseline,repeats=200):
    """Resample whole events/users; preserve all ten dependent candidate rows."""
    y=np.asarray(labels).reshape(-1);n=len(y)//10;event=np.repeat(np.arange(n),10)
    def prepare(p):
        order=np.argsort(np.asarray(p).ravel(),kind='stable');s=np.asarray(p).ravel()[order]
        boundaries=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
        return order,boundaries
    specs=[prepare(p) for p in (prediction,baseline)]
    rng=np.random.default_rng(2026);values=[]
    for _ in range(repeats):
        counts=np.bincount(rng.integers(n,size=n),minlength=n);row_weight=counts[event];aucs=[]
        for order,bounds in specs:
            positive=np.add.reduceat(row_weight[order]*y[order],bounds)
            negative=np.add.reduceat(row_weight[order]*(1-y[order]),bounds)
            aucs.append(float(np.sum(positive*(np.cumsum(negative)-negative/2))/(positive.sum()*negative.sum())))
        values.append([aucs[0],aucs[0]-aucs[1]])
    values=np.array(values)
    return {'method':'paired percentile bootstrap over first events/users; fixed selected recipe; selection uncertainty excluded',
            'repeats':repeats,'auc_95_ci':np.quantile(values[:,0],[.025,.975]).tolist(),
            'gain_vs_official_starter_95_ci':np.quantile(values[:,1],[.025,.975]).tolist()}


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--final-dir',default='competition/final');args=parser.parse_args()
    started=time.monotonic();final=(ROOT/args.final_dir).resolve();manifest=json.loads((final/'manifest.json').read_text())
    version=ROOT/manifest['model_dir'];selection=json.loads((version/'selection.json').read_text())
    assert len(manifest['sources'])==len(manifest['weights']) and abs(sum(manifest['weights'])-1)<1e-12
    for i,name in enumerate(manifest['sources']):
        assert json.loads((version/'models'/str(i)/'recipe.json').read_text())['source_experiment']==name
    light=json.loads((ROOT/'runs/light/delivery_smoke/summary.json').read_text())
    assert light['best_auc']>light['baselines']['user_item_ctr']['auc'] and light['submission_checked']
    assert (ROOT/'overnight.go').is_file() and (ROOT/'LIGHT_START.md').is_file()
    retrieval=[]
    for name in manifest['sources']:
        plan=LAB/'experiments'/name/'retrieval_plan.json'
        assert plan.exists(),f'Missing retrieval evidence for {name}'
        evidence=json.loads(plan.read_text());assert evidence['retrieved']
        retrieval.append({'source':name,'evidence_path':str(plan.relative_to(ROOT)),
                          'retrieved_titles':[r['title'] for r in evidence['retrieved']]})
    readme=(ROOT/'README.md').read_text()
    assert all(name in readme for name in manifest['sources'])
    checks=[]
    commands=[('tests.log',[sys.executable,'-m','unittest','-q','test_arena.py','test_pipeline.py']),
              ('go_vet.log',[str(ROOT/'.tools/go/bin/go'),'vet','overnight.go']),
              ('go_tests.log',[str(ROOT/'.tools/go/bin/go'),'test','overnight.go','overnight_test.go'])]
    for log_name,command in commands:
        run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        (version/log_name).write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError(f'Acceptance command failed: {command}: {run.stderr}')
    _,data=load_cache('data/cache_v1','train')
    for name in dict.fromkeys(manifest['sources']+selection['secondary']+manifest.get('audit_24_sources',[])):
        checks.append(check_experiment(name,data))
    for folder in (LAB/'cache').glob('snapshot_*'):
        if (folder/'metadata.json').exists():
            meta=json.loads((folder/'metadata.json').read_text())
            assert meta['max_fit_timestamp']<meta['cutoff_timestamp']
    submission=ROOT/manifest['submission'];check_submission(submission,'evaluation.zip')
    assert sha256(submission)==manifest['sha256']
    env=dict(os.environ,ARENA_EVALUATION_CACHE_TAG=manifest['sha256'][:16])
    replay=subprocess.run([sys.executable,'-u','arena_finalize.py','--final-dir',str(final),'--predict-only'],cwd=ROOT,env=env,
                          stdout=(version/'reproduction.log').open('w'),stderr=subprocess.STDOUT)
    if replay.returncode:raise RuntimeError('Saved-model replay failed; inspect reproduction.log')
    reproduced=version/'reproduced_submission.csv';maximum=0.;rows=0
    with submission.open(newline='') as left,reproduced.open(newline='') as right:
        a,b=csv.DictReader(left),csv.DictReader(right)
        from itertools import zip_longest
        for x,y in zip_longest(a,b):
            assert x is not None and y is not None and x['rowid']==y['rowid']
            maximum=max(maximum,abs(float(x['predicted_probability'])-float(y['predicted_probability'])));rows+=1
    assert rows==529390 and maximum<=2e-6
    for cutoff in ('26','25','24'):
        v=np.load(version/f'validation_{cutoff}.npz')
        np.testing.assert_array_equal(v['labels'].reshape(-1,10),data['labels'][v['event_indices']])
        assert abs(evaluate(v['labels'],v['predictions'])['auc']-manifest[f'validation_{cutoff}']['auc'])<1e-10
        names=manifest['sources'] if cutoff=='26' else selection['secondary'] if cutoff=='25' else manifest['audit_24_sources']
        records=[prediction_record(json.loads((LAB/'experiments'/name/'result.json').read_text())) for name in names]
        for indices,labels,predictions in records:
            np.testing.assert_array_equal(indices,v['event_indices']);np.testing.assert_array_equal(labels.reshape(-1,10),v['labels'].reshape(-1,10))
        recombined=blend([record[2] for record in records],manifest['weights'],manifest['blend_kind'])
        np.testing.assert_allclose(recombined,v['predictions'],rtol=1e-10,atol=1e-10)
    primary=np.load(version/'validation_26.npz')
    baseline_path=LAB/'experiments/cf_26_fp64_audit/validation.npz'
    if not baseline_path.exists():baseline_path=LAB/'experiments/cf_26/validation.npz'
    starter=np.load(baseline_path)
    np.testing.assert_array_equal(primary['event_indices'],starter['event_indices'])
    baseline=starter['starter'] if 'starter' in starter else starter['predictions']
    ci=clustered_bootstrap(primary['labels'],primary['predictions'],baseline)
    ci['baseline_file']=str(baseline_path.relative_to(ROOT))
    ci['baseline_auc']=float(roc_auc_score(primary['labels'].ravel(),baseline.ravel()))
    artifacts={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha256(p)}
               for base in ('models','representations','source') for p in (version/base).rglob('*') if p.is_file() and not p.name.endswith('.lock')}
    report={'passed':True,'verified_at':now(),'manifest_sha256':sha256(final/'manifest.json'),
            'submission_rows':rows,'submission_sha256':sha256(submission),'reproduction_max_probability_error':maximum,
            'native_experiment_checks':checks,'artifact_hashes':artifacts,'confidence':ci,
            'light_pipeline':{'baseline_auc':light['baselines']['user_item_ctr']['auc'],'trained_auc':light['best_auc'],
                              'submission_checked':light['submission_checked'],'result':'runs/light/delivery_smoke/summary.json'},
            'retrieval_evidence':retrieval,'readme_iterations_recorded':True,
            'completed_experiments':len(list((LAB/'experiments').glob('*/result.json'))),
            'unit_tests_passed':True,'go_vet_passed':True,'go_lane_lock_test_passed':True,'snapshot_temporal_checks_passed':True,
            'official_score_known':False,'seconds':time.monotonic()-started}
    (version/'acceptance.json').write_text(json.dumps(report,indent=2));temporary=final/'acceptance.json.partial'
    temporary.write_text(json.dumps(report,indent=2));os.replace(temporary,final/'acceptance.json')
    note(f"独立验收通过：保存模型和双切点特征重算一致；全部单元测试与Go静态检查通过；{rows:,}行CSV覆盖与概率检查通过；重新加载模型预测最大差异{maximum:.3g}。",'acceptance')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
