"""Export a frozen Qwen checkpoint to the complete competition CSV.

Preparation uses the original environment. Prediction and independent replay
use .venv-generative. The validation-trained head is frozen; evaluation uses
Jan27 past-only representations aligned to the head's Jan24 coordinates.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import subprocess
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from generative import ROOT, LAB, incumbent_digest, load_model, metrics
from generative_data import token_arrays, views_from_config
from arena_report import write_json
from ctr_data import load_cache
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*2**20),b''):digest.update(block)
    return digest.hexdigest()


def write_submission(path,rowids,prediction):
    if rowids.shape!=prediction.shape or prediction.ndim!=2 or prediction.shape[1]!=10:
        raise ValueError('Expected one probability per candidate in every ten-item event')
    if not np.isfinite(prediction).all() or (prediction<0).any() or (prediction>1).any():
        raise ValueError('Invalid submission probabilities')
    temporary=path.with_name(path.stem+'.partial.csv')
    with temporary.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['rowid','predicted_probability'])
        writer.writerows((rid.decode(),f'{float(prob):.10g}') for rid,prob in zip(rowids.ravel(),prediction.ravel()))
    os.replace(temporary,path)


def paths(run):
    return run/'export_inputs',LAB/'generative'/'final'/run.name


def prepare(run):
    from arena_finalize import feature_cache,write_feature_matrix
    from arena_features import complete_features
    before=incumbent_digest()
    config=json.loads((run/'config.json').read_text());dataset=json.loads((run/'dataset.json').read_text())
    with np.load(run/'inputs.npz') as saved:mean=saved['mean'];scale=saved['scale'];selected=saved['valid_indices']
    parent_config=config
    seen=set()
    while parent_config.get('warm_start_run'):
        parent=ROOT/parent_config['warm_start_run']
        if str(parent) in seen:raise ValueError('Warm-start ancestry cycle')
        seen.add(str(parent))
        with np.load(parent/'inputs.npz') as saved:selected=np.union1d(selected,saved['valid_indices'])
        parent_config=json.loads((parent/'config.json').read_text())
    fc=dataset['feature_config'];anchor=fc['train_start'];views=views_from_config(fc)
    inputs,final=paths(run);inputs.mkdir(exist_ok=True);final.mkdir(parents=True,exist_ok=True)
    spec=inputs/'metadata.json'
    if spec.exists():
        old=json.loads(spec.read_text())
        if old['dataset_id']!=dataset['dataset_id']:raise ValueError('Export dataset changed')
        for name,digest in old['input_sha256'].items():
            if sha(inputs/name)!=digest:raise ValueError('Existing export input changed: '+name)
        print('Export inputs already prepared and hashes verified',flush=True);return
    source=json.loads((LAB/'experiments'/config['source_experiment']/'result.json').read_text())
    feature_folder=ROOT/source['feature_dataset'];labels=np.load(feature_folder/'labels.npz')
    meta,train=load_cache('data/cache_v1','train');emeta,evaluation=load_cache('data/cache_v1','evaluation')
    if meta['cache_id']!=dataset['cache_id'] or emeta['cache_id']!=meta['cache_id']:raise ValueError('Data cache identity changed')
    alignment={};counts={}
    for phase,data,indices,cutoff in [
        ('audit',train,labels['validation_indices'],config['cutoff']),
        ('evaluation',evaluation,np.arange(len(evaluation['users'])),'2025-01-27')]:
        if phase=='audit':numeric=np.load(feature_folder/'valid.npy',mmap_mode='r')
        else:numeric=write_feature_matrix(data,indices,cutoff,fc,feature_cache(fc,'evaluation'))
        width=numeric.shape[-1];numeric=numeric.reshape(-1,10,width)
        # Fresh recomputation checks the cached numeric recipe before full export.
        fresh,_=complete_features(data,indices[:8],cutoff,fc)
        np.testing.assert_allclose(fresh.reshape(8,10,width),numeric[:8],rtol=1e-6,atol=1e-6)
        tensors,rotation=token_arrays(data,indices,np.asarray(numeric),cutoff,anchor,views,mean,scale)
        for name,value in tensors.items():
            if not np.isfinite(value).all():raise ValueError('Nonfinite export input')
            np.save(inputs/(phase+'_'+name+'.npy'),value)
        np.save(inputs/(phase+'_indices.npy'),indices)
        if phase=='audit':
            np.save(inputs/'audit_labels.npy',data['labels'][indices])
            np.save(inputs/'audit_heldout.npy',~np.isin(indices,selected))
            manifest=dataset['incumbent_manifest']
            reference=np.load(ROOT/manifest['model_dir']/('validation_'+cutoff[-2:]+'.npz'))
            np.testing.assert_array_equal(reference['event_indices'],indices)
            np.save(inputs/'audit_incumbent.npy',reference['predictions'])
        else:np.save(inputs/'evaluation_rowids.npy',data['rowids'])
        alignment[phase]=rotation;counts[phase]=len(indices)
        print(json.dumps(dict(stage='prepared_export_inputs',phase=phase,events=len(indices))),flush=True)
        del tensors,numeric
    assert incumbent_digest()==before
    metadata=dict(dataset_id=dataset['dataset_id'],cache_id=meta['cache_id'],counts=counts,
        training_cutoff=config['cutoff'],evaluation_representation_cutoff='2025-01-27',anchor=anchor,
        alignment=alignment,input_sha256={p.name:sha(p) for p in sorted(inputs.glob('*.npy'))},
        selection_events_union=len(selected),
        training_scope='Frozen validation-trained head. No final head refit on Jan26 labels; Jan27 behavior representations use all strictly earlier data.',
        incumbent_digest=before)
    write_json(spec,metadata)


def input_arrays(inputs,phase):
    return {key:np.load(inputs/(phase+'_'+key+'.npy'),mmap_mode='r') for key in ('event','history','candidate')}


def load_frozen(run):
    import mlx.core as mx
    config=json.loads((run/'config.json').read_text());source=json.loads((run/'model_source.json').read_text())
    for name,digest in source['weight_sha256'].items():
        if sha(Path(source['local_path'])/name)!=digest:raise ValueError('Base model weight hash changed')
    data=dict(np.load(run/'inputs.npz'));mx.set_cache_limit(256*2**20)
    model,_=load_model(config,data,source);model.load_weights(str(run/'adapters.safetensors'),strict=False)
    model.eval();mx.eval(model.parameters())
    return model,config


def forward(model,arrays,start,end):
    import mlx.core as mx
    value=model.probabilities(*(mx.array(arrays[key][start:end]) for key in ('event','history','candidate')))
    mx.eval(value);result=np.asarray(value)
    if not np.isfinite(result).all() or (result<0).any() or (result>1).any():raise ValueError('Invalid probabilities')
    return result


def predict(run):
    import mlx.core as mx
    inputs,final=paths(run);metadata=json.loads((inputs/'metadata.json').read_text())
    summary=json.loads((run/'summary.json').read_text())
    if (final/'manifest.json').exists():raise ValueError('Completed export retained; use verify to replay it')
    before=incumbent_digest()
    assert before==metadata['incumbent_digest'],'Incumbent changed since export inputs were prepared'
    model,config=load_frozen(run)
    batch=config.get('inference_batch_events',32);started=time.monotonic()
    for phase in ('audit','evaluation'):
        arrays=input_arrays(inputs,phase);n=len(arrays['event']);target=final/(phase+'_predictions.npy')
        progress=final/(phase+'_progress.json');identity=dict(adapter_sha256=sha(run/'adapters.safetensors'),batch_events=batch,
            input_sha256={k:metadata['input_sha256'][phase+'_'+k+'.npy'] for k in arrays})
        done=0
        if target.exists() and progress.exists():
            old=json.loads(progress.read_text())
            if old['identity']!=identity:raise ValueError('Cannot resume predictions with changed model/inputs')
            done=old['completed'];prediction=np.lib.format.open_memmap(target,mode='r+')
        else:prediction=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(n,10))
        for start in range(done,n,batch):
            end=min(start+batch,n);prediction[start:end]=forward(model,arrays,start,end)
            if end%1024==0 or end==n:
                prediction.flush();write_json(progress,dict(identity=identity,completed=end,events=n))
                print(json.dumps(dict(stage='export_prediction',phase=phase,completed=end,events=n,
                    elapsed_seconds=time.monotonic()-started,peak_memory_gib=mx.get_peak_memory()/2**30)),flush=True)
        del prediction,arrays
    labels=np.load(inputs/'audit_labels.npy');p=np.load(final/'audit_predictions.npy')
    reference=np.load(inputs/'audit_incumbent.npy');hold=np.load(inputs/'audit_heldout.npy')
    report=dict(full_time_validation=metrics(labels,p),heldout_time_validation=metrics(labels[hold],p[hold]),
        incumbent_full_time_validation=metrics(labels,reference),incumbent_heldout_time_validation=metrics(labels[hold],reference[hold]),
        selection_cohort=summary['best_validation'],heldout_scope=f"Excludes all {metadata['selection_events_union']} checkpoint-selection events, including warm-start ancestors; same Jan26 first-event-per-user cutoff.",
        training_events=summary['train_events'],trained_event_presentations=summary['trained_event_presentations'])
    write_json(final/'validation.json',report)
    prediction=np.load(final/'evaluation_predictions.npy',mmap_mode='r');rowids=np.load(inputs/'evaluation_rowids.npy',mmap_mode='r')
    write_submission(final/'submission.csv',rowids,prediction)
    assert incumbent_digest()==before
    manifest=dict(method='Qwen3.5-0.8B soft-token LoRA eleven-label classification',model_run=str(run.relative_to(ROOT)),
        model_source=json.loads((run/'model_source.json').read_text()),config=config,
        training_summary_sha256=sha(run/'summary.json'),
        data_audit_sha256=sha(run/'data_audit.json') if (run/'data_audit.json').exists() else None,
        adapter_sha256=sha(run/'adapters.safetensors'),export_input_metadata_sha256=sha(inputs/'metadata.json'),
        csv=str((final/'submission.csv').relative_to(ROOT)),rows=int(prediction.size),csv_sha256=sha(final/'submission.csv'),
        prediction_sha256={phase:sha(final/(phase+'_predictions.npy')) for phase in ('audit','evaluation')},
        source_sha256={name:sha(ROOT/name) for name in ('generative.py','generative_data.py','generative_model.py','generative_export.py')},
        validation=report,training_scope=metadata['training_scope'],incumbent_digest=before,
        promoted=False,seconds=time.monotonic()-started)
    write_json(final/'manifest.json',manifest)
    import shutil
    shutil.copy2(ROOT/'generative_export.py',run/'source'/'generative_export.py')
    print(json.dumps(dict(stage='csv_exported',csv=manifest['csv'],rows=manifest['rows'],**report),indent=2),flush=True)


def verify(run):
    from run import check_submission
    inputs,final=paths(run);manifest=json.loads((final/'manifest.json').read_text())
    metadata=json.loads((inputs/'metadata.json').read_text())
    assert incumbent_digest()==manifest['incumbent_digest']
    assert sha(final/'submission.csv')==manifest['csv_sha256']
    assert sha(run/'adapters.safetensors')==manifest['adapter_sha256']
    assert sha(run/'summary.json')==manifest['training_summary_sha256']
    if manifest.get('data_audit_sha256'):
        assert sha(run/'data_audit.json')==manifest['data_audit_sha256']
        assert json.loads((run/'data_audit.json').read_text())['passed']
    assert sha(inputs/'metadata.json')==manifest['export_input_metadata_sha256']
    for name,digest in manifest['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in metadata['input_sha256'].items():assert sha(inputs/name)==digest,name
    for phase,digest in manifest['prediction_sha256'].items():assert sha(final/(phase+'_predictions.npy'))==digest
    model,config=load_frozen(run);batch=config.get('inference_batch_events',32);errors=[]
    for phase in ('audit','evaluation'):
        arrays=input_arrays(inputs,phase);n=len(arrays['event']);expected=np.load(final/(phase+'_predictions.npy'),mmap_mode='r')
        # Replay the exact original batch boundaries, including the short tail.
        starts=sorted(set([0,17*batch,(n//2//batch)*batch,((n-1)//batch)*batch]))
        for start in starts:
            end=min(start+batch,n);actual=forward(model,arrays,start,end)
            error=float(np.max(np.abs(actual-expected[start:end])));errors.append(error)
            if error>2e-5:raise ValueError('Independent checkpoint replay differs')
    check_submission(final/'submission.csv','evaluation.zip')
    expected=np.load(final/'evaluation_predictions.npy',mmap_mode='r').ravel()
    rowids=np.load(inputs/'evaluation_rowids.npy',mmap_mode='r').ravel();count=0;rounding_error=0.
    with (final/'submission.csv').open(newline='') as handle:
        reader=csv.reader(handle);next(reader)
        for rid,raw in reader:
            if rid!=rowids[count].decode():raise ValueError('CSV row order/identity differs from evaluation cache')
            rounding_error=max(rounding_error,abs(float(raw)-float(expected[count])));count+=1
    assert count==len(expected)==529390 and rounding_error<1e-9
    assert incumbent_digest()==manifest['incumbent_digest']
    result=dict(passed=True,rows=count,csv_sha256=sha(final/'submission.csv'),manifest_sha256=sha(final/'manifest.json'),
        fresh_process_model_replay_max_error=max(errors),csv_probability_rounding_max_error=rounding_error,
        export_input_hashes_checked=len(metadata['input_sha256']),incumbent_unchanged=True)
    write_json(final/'acceptance.json',result)
    from arena_report import note
    note(f"生成式训练与提交完成：{run.name}，完整 CSV {count:,} 行，Jan26 完整时间验证 AUC {manifest['validation']['full_time_validation']['auc']:.6f}；独立重载验收通过，原最优保留。",'generative')
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','predict','verify','finish']);p.add_argument('--run',required=True)
    a=p.parse_args();run=(ROOT/a.run).resolve()
    if a.command=='finish':
        # The active trainer owns this lock until its durable summary is complete.
        with (run/'.training.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_SH)
        if not (run/'summary.json').exists():raise ValueError('Training stopped without a completed checkpoint summary')
        base=Path(sys.base_prefix)/'bin/python3';venv=ROOT/'.venv-generative/bin/python'
        subprocess.run([str(venv),str(ROOT/'generative.py'),'replay','--run',str(run)],cwd=ROOT,check=True)
        subprocess.run([str(base),str(ROOT/'generative_export.py'),'prepare','--run',str(run)],cwd=ROOT,check=True)
        _,final=paths(run)
        if not (final/'manifest.json').exists():
            subprocess.run([str(venv),str(ROOT/'generative_export.py'),'predict','--run',str(run)],cwd=ROOT,check=True)
        subprocess.run([str(venv),str(ROOT/'generative_export.py'),'verify','--run',str(run)],cwd=ROOT,check=True)
        return
    {'prepare':prepare,'predict':predict,'verify':verify}[a.command](run)


if __name__=='__main__':main()
