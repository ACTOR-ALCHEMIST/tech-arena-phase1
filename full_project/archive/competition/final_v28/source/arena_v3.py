"""Three reproducible upgrades with larger history Transformers and real HF bases.

Global Python runs the Torch/Transformers MPS pipeline. Existing best files are
read-only. All representation snapshots precede their respective query cutoff.
"""
import os
for k,v in {'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
    'PYTORCH_MPS_HIGH_WATERMARK_RATIO':'.55','PYTORCH_MPS_LOW_WATERMARK_RATIO':'.4',
    'TOKENIZERS_PARALLELISM':'false'}.items():os.environ.setdefault(k,v)
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
from arena_report import write_json

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'competition/v3'
SOURCE='v2_multiview_wide256_s2026'
RECIPES={
 'r1_history128':dict(width=128,heads=4,history_layers=2,slate_layers=1,hidden=[512,256,128],context_width=64,foundation_dims=0,foundation=None),
 'r2_qwen3_history128':dict(width=128,heads=4,history_layers=2,slate_layers=1,hidden=[512,256,128],context_width=64,foundation_dims=32,foundation='Qwen3-0.6B'),
 'r3_qwen25_history192':dict(width=192,heads=6,history_layers=3,slate_layers=2,hidden=[768,384,192],context_width=96,foundation_dims=32,foundation='Qwen2.5-1.5B',foundation_layers=6,unfreeze_base=True)}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def guard():return {n:sha(ROOT/'competition/final'/n) for n in ['submission.csv','manifest.json','acceptance.json']}


def base_checkpoint(cutoff):
    if cutoff=='2025-01-27':
        m=json.loads((ROOT/'competition/final/manifest.json').read_text());i=m['sources'].index(SOURCE)
        return ROOT/m['model_dir']/'models'/str(i)/'model.pt'
    return ROOT/'competition/experiments'/(SOURCE if cutoff=='2025-01-26' else SOURCE+'_locked'+cutoff[-2:])/'model.pt'


def prepare(cutoff):
    import torch
    from ctr_data import load_cache
    from generative_data import token_arrays,views_from_config
    from arena_models import feature_dataset
    from arena_features import complete_features
    from run import timestamp
    from datetime import datetime,timedelta
    folder=OUT/'data'/cutoff;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metadata.json').exists():return folder
    before=guard();source=json.loads((ROOT/'competition/experiments'/SOURCE/'result.json').read_text())
    fc=copy.deepcopy(source['feature_config'])
    if cutoff!='2025-01-27':
        delta=datetime.fromisoformat('2025-01-26')-datetime.fromisoformat(cutoff)
        fc['train_start']=(datetime.fromisoformat(fc['train_start'])-delta).date().isoformat()
    anchor=fc['train_start'];meta,data=load_cache('data/cache_v1','train')
    checkpoint=base_checkpoint(cutoff);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    mean=saved['state_dict']['mean'].numpy();scale=saved['state_dict']['scale'].numpy()
    np.savez(folder/'normalizer.npz',mean=mean,scale=scale)
    if cutoff=='2025-01-27':
        from arena_finalize import write_feature_matrix,feature_cache
        fc['train_start']=anchor
        ti=np.flatnonzero((data['timestamps']>=timestamp(anchor))&(data['timestamps']<timestamp(cutoff)))
        # Dedicated key, so existing refit caches are never replaced.
        x=write_feature_matrix(data,ti,anchor,fc,folder/'numeric_train.npy')
        _,evaluation=load_cache('data/cache_v1','evaluation');vi=np.arange(len(evaluation['users']))
        v=write_feature_matrix(evaluation,vi,cutoff,fc,feature_cache(fc,'evaluation'))
        phases=[('train',data,ti,x,anchor),('valid',evaluation,vi,v,cutoff)]
    else:
        cached=feature_dataset(cutoff,anchor,180000,42,fc['latent_factors'],True,fc['sgns_factors'],fc['sgns_epochs'],fc['sgns_recipes'],sgns_extra=fc['sgns_extra'])
        labels=np.load(cached/'labels.npz');ti=labels['train_indices'];vi=labels['validation_indices']
        x=np.load(cached/'train.npy',mmap_mode='r');v=np.load(cached/'valid.npy',mmap_mode='r')
        phases=[('train',data,ti,x,anchor),('valid',data,vi,v,cutoff)]
    context_mask=(np.asarray(data['context'][ti]).std(0)>1e-4).astype(np.float32)
    alignment={}
    for phase,raw,indices,numeric,representation_cutoff in phases:
        numeric=numeric.reshape(-1,10,112)
        arrays,rotation=token_arrays(raw,indices,numeric,representation_cutoff,anchor,views_from_config(fc),mean,scale)
        arrays['event'][:,32:38]*=context_mask
        for key,value in arrays.items():
            assert np.isfinite(value).all();np.save(folder/(phase+'_'+key+'.npy'),value)
        np.save(folder/(phase+'_indices.npy'),indices)
        if phase=='train' or cutoff!='2025-01-27':np.save(folder/(phase+'_labels.npy'),raw['labels'][indices])
        else:np.save(folder/'rowids.npy',raw['rowids'])
        alignment[phase]=rotation
        print(json.dumps(dict(stage='data_prepared',cutoff=cutoff,phase=phase,events=len(indices))),flush=True)
        del arrays
    if cutoff!='2025-01-27':
        manifest=json.loads((ROOT/'competition/final/manifest.json').read_text());b=np.load(ROOT/manifest['model_dir']/('validation_'+cutoff[-2:]+'.npz'))
        np.testing.assert_array_equal(b['event_indices'],vi);np.save(folder/'incumbent.npy',b['predictions'])
    assert guard()==before and int(data['timestamps'][ti].max())<timestamp(cutoff)
    write_json(folder/'metadata.json',dict(cutoff=cutoff,anchor=anchor,cache_id=meta['cache_id'],train_events=len(ti),valid_events=len(vi),
        base_checkpoint=str(checkpoint.relative_to(ROOT)),base_checkpoint_sha256=sha(checkpoint),base_config=saved['config'],
        feature_config=fc,alignment=alignment,context_mask=context_mask.tolist(),incumbent=before,
        max_train_timestamp=int(data['timestamps'][ti].max()),training_representation_cutoff=anchor))
    return folder


def arrays(folder,phase,foundation=None):
    values={k:np.load(folder/(phase+'_'+k+'.npy'),mmap_mode='r') for k in ['event','history','candidate']}
    if foundation:values['foundation']=np.load(folder/(phase+'_'+foundation+'.npy'),mmap_mode='r')
    return values


def tensors(values,index,device):
    import torch
    return [torch.as_tensor(np.array(v[index]),device=device) for v in values.values()]


def predict(model,values,batch=512):
    import torch
    model.eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(values['event']),batch):
            result.append(model.probabilities(*tensors(values,slice(start,start+batch),'mps')).cpu().numpy())
    return np.concatenate(result)


def train(name,cutoff='2025-01-26',fixed_epochs=None):
    import torch
    from torch.nn import functional as F
    from arena_v3_model import HistoryTransformer
    from arena_core import evaluate
    import fcntl
    torch.set_num_threads(2);torch.manual_seed(2026)
    folder=prepare(cutoff);meta=json.loads((folder/'metadata.json').read_text());before=guard()
    out=OUT/'runs'/name/cutoff;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
    with (out/'.training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=dict(RECIPES[name],name=name,seed=2026,dropout=.1,epochs=12,lr=.0004,batch_events=256,patience=3,base_config=meta['base_config'])
        train_values=arrays(folder,'train',config['foundation']);valid_values=arrays(folder,'valid',config['foundation'])
        y=np.load(folder/'train_labels.npy');target=np.where(y.sum(1)>0,y.argmax(1),10)
        model=HistoryTransformer(config)
        saved=torch.load(ROOT/meta['base_checkpoint'],map_location='cpu',weights_only=True)
        state=saved['state_dict'];state['mean']=torch.zeros(112);state['scale']=torch.ones(112);model.base.load_state_dict(state)
        if config['foundation']:
            f=np.asarray(train_values['foundation']);model.foundation_mean.copy_(torch.from_numpy(f.mean((0,1))))
            model.foundation_scale.copy_(torch.from_numpy(f.std((0,1)).clip(1e-3)))
        model.to('mps')
        groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('base.')],lr=config['lr'])]
        if config.get('unfreeze_base'):groups.append(dict(params=model.base.parameters(),lr=.00003))
        optimizer=torch.optim.AdamW(groups,weight_decay=.03)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'],eta_min=.00003)
        initial=predict(model,valid_values);validation_labels=None if cutoff=='2025-01-27' else np.load(folder/'valid_labels.npy')
        best=evaluate(validation_labels,initial)['auc'] if validation_labels is not None else -1
        write_json(out/'config.json',config);records=[];rng=np.random.default_rng(2026);stale=0;best_epoch=0;started=time.monotonic()
        def save(p):
            torch.save(dict(state_dict=model.state_dict(),config=config),out/'model.pt')
            np.savez_compressed(out/'validation.npz',event_indices=np.load(folder/'valid_indices.npy'),predictions=p,
                **({'labels':validation_labels} if validation_labels is not None else {}))
        save(initial)
        print(json.dumps(dict(stage='initial',name=name,cutoff=cutoff,auc=best,parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))),flush=True)
        epochs=fixed_epochs if fixed_epochs is not None else config['epochs']
        for epoch in range(1,epochs+1):
            tick=time.monotonic();model.train();order=rng.permutation(len(target));total=0.
            for offset in range(0,len(order),config['batch_events']):
                index=order[offset:offset+config['batch_events']];inputs=tensors(train_values,index,'mps')
                labels=torch.as_tensor(target[index],device='mps',dtype=torch.long)
                loss=F.cross_entropy(model(*inputs),labels)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                total+=float(loss.item())*len(index)
            scheduler.step();row=dict(epoch=epoch,train_loss=total/len(target),seconds=time.monotonic()-tick)
            if validation_labels is not None:
                p=predict(model,valid_values);row.update(evaluate(validation_labels,p))
                improved=row['auc']>best+1e-7
                if fixed_epochs is not None or improved:
                    best=row['auc'];best_epoch=epoch;stale=0;save(p)
                else:stale+=1
            elif epoch==epochs:save(predict(model,valid_values));best_epoch=epoch
            records.append(row);write_json(out/'training.json',records);print(json.dumps(dict(name=name,cutoff=cutoff,**row)),flush=True)
            if fixed_epochs is None and stale>=config['patience']:break
        checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict']);replayed=predict(model,valid_values)
        expected=np.load(out/'validation.npz')['predictions'];error=float(np.max(np.abs(replayed-expected)));assert error<2e-5
        assert guard()==before
        result=dict(name=name,cutoff=cutoff,config=config,epochs=records,best_epoch=best_epoch,auc=best,
            train_events=len(target),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total_head_parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-started,
            native_reload_max_error=error,incumbent_unchanged=True,model_sha256=sha(out/'model.pt'),data=str(folder.relative_to(ROOT)))
        write_json(out/'result.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);p.add_argument('--cutoff',default='2025-01-26')
    p.add_argument('--name',choices=list(RECIPES),default='r1_history128');p.add_argument('--fixed-epochs',type=int)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare':prepare(a.cutoff)
    else:train(a.name,a.cutoff,a.fixed_epochs)


if __name__=='__main__':main()
