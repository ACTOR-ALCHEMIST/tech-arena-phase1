"""Independent LLM experiment: prepare -> train -> reload/replay.

Use .venv-generative/bin/python. Existing baseline exports are read-only.
This runner intentionally has no automatic promotion to competition/final.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('USE_TORCH','0')
os.environ.setdefault('USE_TF','0')
import numpy as np
from arena_report import write_json
from generative_data import permute_candidates

ROOT=Path(__file__).resolve().parent
LAB=ROOT/'competition'


def metrics(y,p):
    from sklearn.metrics import roc_auc_score,log_loss
    if not np.isfinite(p).all():raise ValueError('Non-finite prediction')
    return dict(auc=float(roc_auc_score(np.asarray(y).ravel(),p.ravel())),
                logloss=float(log_loss(np.asarray(y).ravel(),p.ravel())),events=len(p))


def incumbent_digest():
    return {name:hashlib.sha256((LAB/'final'/name).read_bytes()).hexdigest()
            for name in ['manifest.json','submission.csv','acceptance.json']}


def locate_data(config):
    index=LAB/'generative'/(config['name']+'_data.json')
    if not index.exists():
        base=Path(sys.base_prefix)/'bin/python3'
        subprocess.run([str(base),str(ROOT/'generative_data.py'),'--config',config['_config_path']],cwd=ROOT,check=True)
    saved=json.loads(index.read_text())
    for key in ['source_experiment','cutoff','train_events','valid_events','split_seed']:
        if config[key]!=saved['config'][key]:raise ValueError(f'Dataset differs at {key}; rerun prepare with this config')
    return ROOT/saved['dataset']


def load_model(config,data,source=None):
    import mlx.core as mx
    from mlx_lm import load
    from generative_model import GenerativeCTR
    source=source or json.loads((LAB/'generative/model_source.json').read_text())
    if source['repo_id']!=config['model_id'] or source['revision']!=config['revision']:
        raise ValueError('Model identity/revision differs; download the requested model with setup_generative.py')
    mx.random.seed(config['seed'])
    backbone,tokenizer=load(source['local_path'])
    dimensions={k:data['train_'+k].shape[-1] for k in ('event','history','candidate')}
    counts=np.bincount(data['train_target'],minlength=11)+1
    model=GenerativeCTR(backbone,tokenizer,dimensions,config,(counts/counts.sum()).astype(np.float32))
    return model,source


def predict(model,data,prefix='valid',batch_size=1,logits=False,limit=None):
    import mlx.core as mx
    model.eval();parts=[];n=len(data[prefix+'_event'])
    if limit is not None:n=min(limit,n)
    for start in range(0,n,batch_size):
        tensors=[mx.array(data[prefix+'_'+k][start:start+batch_size]) for k in ('event','history','candidate')]
        z=model(*tensors);value=z if logits else mx.softmax(z,axis=-1)[:,:10]
        mx.eval(value);parts.append(np.asarray(value))
    return np.concatenate(parts)


def probe_loss(model,data):
    z=predict(model,data,'train',logits=True,limit=8)
    z=z-z.max(1,keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(1))-z[np.arange(len(z)),data['train_target'][:len(z)]]))


def save_trainable(model,path):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    temporary=path.with_name(path.stem+'.partial.safetensors')
    values=dict(tree_flatten(model.trainable_parameters()))
    mx.eval(values);mx.save_safetensors(str(temporary),values);os.replace(temporary,path)


def save_provenance(out):
    import shutil
    import platform
    from importlib.metadata import version
    source=out/'source';source.mkdir(exist_ok=True)
    for name in ['generative.py','generative_data.py','generative_model.py','setup_generative.py',
                 'test_generative.py','test_generative_model.py','requirements-generative.txt']:
        shutil.copy2(ROOT/name,source/name)
    write_json(out/'environment.json',dict(python=sys.version,platform=platform.platform(),
        packages={p:version(p) for p in ['mlx','mlx-lm','mlx-metal','transformers','huggingface-hub','numpy','scipy','scikit-learn']}))


def train(config):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten,tree_map
    started=time.monotonic();before=incumbent_digest()
    folder=locate_data(config);data=dict(np.load(folder/'inputs.npz'));metadata=json.loads((folder/'metadata.json').read_text())
    out=LAB/'generative'/'runs'/config['name'];out.mkdir(parents=True,exist_ok=True)
    with (out/'.training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'summary.json').exists():raise ValueError('Completed run retained. Use a new config name for a new experiment.')
        mx.set_cache_limit(256*2**20)
        model,source=load_model(config,data);mx.eval(model.parameters())
        dimensions={k:data['train_'+k].shape[-1] for k in ('event','history','candidate')}
        trainable={n:int(v.size) for n,v in tree_flatten(model.trainable_parameters())}
        print(json.dumps({'device':mx.device_info()['device_name'],'tokens':metadata['token_count'],
                          'trainable_parameters':sum(trainable.values()),'lora_tensors':sum(n.startswith('backbone.') for n in trainable)},indent=2),flush=True)
        # Durable small dataset and provenance, independent of feature-cache cleanup.
        np.savez_compressed(out/'inputs.npz',**data)
        write_json(out/'dataset.json',metadata);write_json(out/'config.json',config);write_json(out/'model_source.json',source)
        write_json(out/'trainable_parameters.json',trainable)
        initial=predict(model,data,batch_size=config['batch_events'])
        initial_metrics=metrics(data['valid_labels'],initial);initial_probe=probe_loss(model,data)
        print(json.dumps({'stage':'before_training',**initial_metrics,'train_probe_loss':initial_probe}),flush=True)
        np.savez_compressed(out/'initial_validation.npz',event_indices=data['valid_indices'],labels=data['valid_labels'],predictions=initial)
        optimizer=optim.AdamW(learning_rate=config['learning_rate'],weight_decay=.01)
        def objective(m,event,history,candidate,target):
            return nn.losses.cross_entropy(m(event,history,candidate),target,reduction='mean')
        loss_and_grad=nn.value_and_grad(model,objective)
        rng=np.random.default_rng(config['seed']);order=rng.permutation(len(data['train_target']));position=0
        records=[];best=-1;best_step=0;trained_events=0
        for step in range(1,config['steps']+1):
            tick=time.monotonic();model.train()
            gradients=None;loss_sum=mx.array(0.);step_events=0
            for _ in range(config.get('gradient_accumulation_steps',1)):
                if position>=len(order):order=rng.permutation(len(order));position=0
                chosen=order[position:position+config['batch_events']];position+=len(chosen)
                candidate=data['train_candidate'][chosen];target=data['train_target'][chosen]
                if config.get('candidate_shuffle',True):candidate,target,_=permute_candidates(candidate,target,rng)
                values=[mx.array(data['train_event'][chosen]),mx.array(data['train_history'][chosen]),mx.array(candidate),mx.array(target)]
                micro_loss,micro_grad=loss_and_grad(model,*values)
                micro_grad=tree_map(lambda g:g*len(chosen),micro_grad)
                gradients=micro_grad if gradients is None else tree_map(lambda a,b:a+b,gradients,micro_grad)
                loss_sum=loss_sum+micro_loss*len(chosen);step_events+=len(chosen)
                mx.eval(loss_sum,gradients)
            loss=loss_sum/step_events;gradients=tree_map(lambda g:g/step_events,gradients)
            norm=mx.sqrt(sum(mx.sum(g.astype(mx.float32)**2) for _,g in tree_flatten(gradients)))
            factor=mx.minimum(1.,1./mx.maximum(norm,1e-8))
            gradients=tree_map(lambda g:g*factor,gradients)
            optimizer.update(model,gradients);mx.eval(loss,norm,model.trainable_parameters(),optimizer.state)
            if not bool(mx.isfinite(loss)) or not bool(mx.isfinite(norm)):raise ValueError('Non-finite loss/gradient')
            trained_events+=step_events
            record=dict(step=step,loss=float(loss),gradient_norm=float(norm),seconds=time.monotonic()-tick,
                        peak_memory_gib=mx.get_peak_memory()/2**30,trained_event_presentations=trained_events)
            records.append(record);print(json.dumps(record),flush=True)
            if step%config['eval_every']==0 or step==config['steps']:
                p=predict(model,data,batch_size=config['batch_events']);record['validation']=metrics(data['valid_labels'],p)
                print(json.dumps({'stage':'validation','step':step,**record['validation']}),flush=True)
                if record['validation']['auc']>best:
                    best=record['validation']['auc'];best_step=step;save_trainable(model,out/'adapters.safetensors')
                    np.savez_compressed(out/'validation.npz',event_indices=data['valid_indices'],labels=data['valid_labels'],predictions=p)
            write_json(out/'training.json',records)
        final_probe=probe_loss(model,data)
        model.load_weights(str(out/'adapters.safetensors'),strict=False)
        reloaded=predict(model,data,batch_size=config['batch_events'])
        expected=np.load(out/'validation.npz')['predictions'];error=float(np.max(np.abs(reloaded-expected)))
        if error>2e-5:raise ValueError('Saved adapter replay differs')
        assert incumbent_digest()==before,'Incumbent files changed during independent experiment'
        seconds=time.monotonic()-started;step_seconds=float(np.median([r['seconds'] for r in records[1:]] or [r['seconds'] for r in records]))
        result=dict(name=config['name'],method='qwen35_softtoken_label_classification',cutoff=config['cutoff'],
            initial_validation=initial_metrics,best_validation=metrics(data['valid_labels'],reloaded),best_step=best_step,
            baseline_same_cohort=metadata['baseline'],validation_id=metadata['validation_id'],
            train_events=len(data['train_target']),valid_events=len(data['valid_target']),steps=config['steps'],
            trained_event_presentations=trained_events,initial_train_probe_loss=initial_probe,final_train_probe_loss=final_probe,
            trainable_parameters=sum(trainable.values()),tokens_per_event=metadata['token_count'],
            peak_memory_gib=mx.get_peak_memory()/2**30,median_step_seconds=step_seconds,
            estimated_20000_event_epoch_seconds=step_seconds*20000/(config['batch_events']*config.get('gradient_accumulation_steps',1)),
            estimates_scope='Linear estimate from this microbenchmark, excluding validation, first-step compilation and long-run thermal effects.',
            adapter_replay_max_error=error,incumbent_unchanged=True,incumbent_sha256=before['submission.csv'],
            seconds=seconds,scope='Smoke experiment, not a full-training or official leaderboard result. Selection on this small cohort may overfit.')
        write_json(out/'summary.json',result)
        save_provenance(out)
        print(json.dumps(result,indent=2),flush=True)
        from arena_report import note
        note(f"生成式轻量分支 {config['name']}：Qwen3.5真实训练 {config['steps']} 步，峰值 {result['peak_memory_gib']:.2f} GiB，同队列 AUC {best:.6f}，当前最优提交保留。",'generative')
        return result


def replay(folder):
    import mlx.core as mx
    before=incumbent_digest();config=json.loads((folder/'config.json').read_text())
    data=dict(np.load(folder/'inputs.npz'));source=json.loads((folder/'model_source.json').read_text())
    mx.set_cache_limit(256*2**20);model,_=load_model(config,data,source)
    model.load_weights(str(folder/'adapters.safetensors'),strict=False)
    p=predict(model,data,batch_size=config['batch_events']);expected=np.load(folder/'validation.npz')['predictions']
    error=float(np.max(np.abs(p-expected)))
    assert error<=2e-5 and incumbent_digest()==before
    write_json(folder/'replay.json',dict(passed=True,max_probability_error=error,**metrics(data['valid_labels'],p)))
    print(json.dumps(dict(passed=True,max_probability_error=error,events=len(p)),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','train','replay'])
    parser.add_argument('--config',default='configs/generative_smoke.json');parser.add_argument('--run');args=parser.parse_args()
    if args.command=='replay':
        if not args.run:parser.error('--run is required for replay')
        replay(ROOT/args.run);return
    path=(ROOT/args.config).resolve();config=json.loads(path.read_text());config['_config_path']=str(path)
    if args.command=='prepare':
        subprocess.run([str(Path(sys.base_prefix)/'bin/python3'),str(ROOT/'generative_data.py'),'--config',str(path)],cwd=ROOT,check=True)
    else:train(config)


if __name__=='__main__':main()
