"""Past-only soft-token inputs. No language-model dependencies or opaque ID text.

Use the original Python environment to prepare inputs; the isolated MLX
environment only loads these small arrays. Validation item coordinates are
aligned to the strictly earlier training snapshot, without validation labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from arena_report import write_json
from ctr_data import load_cache

ROOT=Path(__file__).resolve().parent
LAB=ROOT/'competition'


def class_targets(labels):
    labels=np.asarray(labels).reshape(-1,10)
    if not np.isin(labels,[0,1]).all() or (labels.sum(1)>1).any():
        raise ValueError('Expected ten binary outcomes and at most one click')
    return np.where(labels.sum(1)>0,labels.argmax(1),10).astype(np.int32)


def permute_candidates(candidate, targets, rng):
    """Each row is new-slot -> old-slot; class 10 (no click) stays class 10."""
    order=np.argsort(rng.random(candidate.shape[:2]),axis=1)
    shuffled=np.take_along_axis(candidate,order[:,:,None],axis=1)
    mapped=np.where(targets==10,10,(order==targets[:,None]).argmax(1)).astype(np.int32)
    return shuffled,mapped,order


def restore_probabilities(probability, order):
    return np.take_along_axis(probability,np.argsort(order,axis=1),axis=1)


def normalize_rows(x):
    return (x/np.linalg.norm(x,axis=-1,keepdims=True).clip(1e-6)).astype(np.float32)


def views_from_config(fc):
    views=[dict(factors=f,epochs=fc.get('sgns_epochs',{}).get(str(f),80 if f<=8 else 56),
                recipe=fc.get('sgns_recipes',{}).get(str(f))) for f in fc.get('sgns_factors',[])]
    views.extend(dict(factors=s['factors'],epochs=s['recipe']['epochs'],recipe=s['recipe']) for s in fc.get('sgns_extra',[]))
    return views


def behavioral_arrays(data,indices,cutoff,anchor,views):
    from arena_core import snapshot
    from arena_sgns_features import embedding_model
    from arena_alignment import orthogonal_map
    users=[];items=[];histories=[];alignment=[]
    stats=dict(np.load(snapshot(cutoff)/'stats.npz'))
    past=dict(np.load(snapshot(anchor)/'stats.npz'))
    for spec in views:
        factors=embedding_model(cutoff,**spec)
        d=spec['factors'];rotation=np.eye(d,dtype=np.float32)
        if cutoff!=anchor:
            reference=embedding_model(anchor,**spec)
            keep=(past['pop']>=3)&(stats['pop']>=3)
            if keep.sum()<d:raise ValueError('Too few past anchors for coordinate alignment')
            rotation=orthogonal_map(normalize_rows(factors['items'][keep]),normalize_rows(reference['items'][keep]))
        u=normalize_rows(factors['users']@rotation);v=normalize_rows(factors['items']@rotation)
        u[~stats['known_users']]=0;v[stats['pop']==0]=0;v[0]=0
        users.append(u[data['users'][indices]])
        items.append(v[data['items'][indices]])
        histories.append(v[data['history'][indices]])
        alignment.append(dict(spec=spec,rotation=rotation.tolist(),anchor=anchor,cutoff=cutoff))
    return tuple(np.concatenate(parts,axis=-1) for parts in (users,items,histories)),alignment


def token_arrays(data,indices,numeric,cutoff,anchor,views,mean,scale):
    (users,items,hist),alignment=behavioral_arrays(data,indices,cutoff,anchor,views)
    numeric=np.clip((numeric-mean)/scale,-6,6).astype(np.float32)
    valid=data['history'][indices]!=0;ages=np.asarray(data['age_hours'][indices])
    now=np.asarray(data['timestamps'][indices]);previous=now[:,None]-np.rint(ages*3600).astype(np.int64)
    same_bucket=(previous%86400//21600)==(now%86400//21600)[:,None]
    same_weekend=(((previous//86400+3)%7)>=5)==(((now//86400+3)%7)>=5)[:,None]
    temporal=np.stack([np.log1p(ages)/np.log1p(720),same_bucket,same_weekend,valid],axis=-1)
    temporal*=valid[:,:,None]
    history=np.concatenate([hist,temporal],axis=-1).astype(np.float32)
    candidate=np.concatenate([numeric,items],axis=-1).astype(np.float32)
    aggregate=np.concatenate([numeric.mean(1),numeric.max(1),numeric.std(1)],axis=-1)
    event=np.concatenate([users,np.asarray(data['context'][indices]),aggregate],axis=-1).astype(np.float32)
    return dict(event=event,history=history,candidate=candidate),alignment


def prepare(config):
    from arena_core import evaluate
    source=json.loads((LAB/'experiments'/config['source_experiment']/'result.json').read_text())
    if source['cutoff']!=config['cutoff']:raise ValueError('Use a source experiment at the same validation cutoff')
    fc=source['feature_config'];folder=ROOT/source['feature_dataset'];labels=np.load(folder/'labels.npz')
    _,data=load_cache('data/cache_v1','train');meta,_=load_cache('data/cache_v1','train')
    anchor=fc['train_start'];views=views_from_config(fc)
    if not views:raise ValueError('Source has no behavior embeddings')
    # Independent sampling keeps the same validation cohort when training grows.
    tr_rng=np.random.default_rng(config['split_seed']);va_rng=np.random.default_rng(config['split_seed']+1)
    ti=labels['train_indices'];vi=labels['validation_indices']
    tr=np.sort(tr_rng.choice(len(ti),min(config['train_events'] or len(ti),len(ti)),replace=False))
    va=np.sort(va_rng.choice(len(vi),min(config['valid_events'] or len(vi),len(vi)),replace=False))
    x=np.load(folder/'train.npy',mmap_mode='r');v=np.load(folder/'valid.npy',mmap_mode='r')
    selected=np.asarray(x.reshape(-1,10,x.shape[1])[tr])
    mean=selected.mean((0,1));std=selected.std((0,1));scale=np.where(std<1e-4,1e12,std).astype(np.float32)
    normalizer=None
    if config.get('normalizer_run'):
        parent=ROOT/config['normalizer_run'];parent_meta=json.loads((parent/'dataset.json').read_text())
        if parent_meta['feature_config']!=fc:raise ValueError('Warm-start feature recipe differs')
        with np.load(parent/'inputs.npz') as saved:mean=saved['mean'];scale=saved['scale']
        normalizer=dict(run=config['normalizer_run'],sha256=hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest())
    train,train_alignment=token_arrays(data,ti[tr],selected,anchor,anchor,views,mean,scale)
    valid,valid_alignment=token_arrays(data,vi[va],np.asarray(v.reshape(-1,10,v.shape[1])[va]),config['cutoff'],anchor,views,mean,scale)
    arrays={**{'train_'+k:a for k,a in train.items()},**{'valid_'+k:a for k,a in valid.items()},
            'train_target':class_targets(data['labels'][ti[tr]]),'valid_target':class_targets(data['labels'][vi[va]]),
            'train_indices':ti[tr],'valid_indices':vi[va],'valid_labels':data['labels'][vi[va]],'mean':mean,'scale':scale}
    current=json.loads((LAB/'final/manifest.json').read_text())
    baseline=np.load(ROOT/current['model_dir']/('validation_'+config['cutoff'][-2:]+'.npz'))
    assert np.array_equal(vi,baseline['event_indices'])
    arrays['baseline_prediction']=baseline['predictions'][va]
    digest=hashlib.sha256(meta['cache_id'].encode());digest.update(ti[tr].tobytes());digest.update(vi[va].tobytes())
    digest.update(json.dumps(fc,sort_keys=True).encode())
    digest.update(mean.tobytes()+scale.tobytes())
    valid_digest=hashlib.sha256(meta['cache_id'].encode()+vi[va].tobytes()).hexdigest()
    out=LAB/'generative'/'data'/digest.hexdigest()[:16];out.mkdir(parents=True,exist_ok=True)
    import os
    temporary=out/f'inputs.{os.getpid()}.partial.npz'
    np.savez_compressed(temporary,**arrays);os.replace(temporary,out/'inputs.npz')
    metadata=dict(config=config,dataset_id=digest.hexdigest(),validation_id=valid_digest,cache_id=meta['cache_id'],
                  train_events=len(tr),validation_events=len(va),source_experiment=source['name'],feature_config=fc,
                  training_representation_cutoff=anchor,validation_representation_cutoff=config['cutoff'],
                  token_count=32,embedding_dimensions=sum(s['factors'] for s in views),
                  shapes={k:list(a.shape) for k,a in arrays.items()},views=views,
                  normalizer=normalizer,
                  alignment={'train':train_alignment,'valid':valid_alignment},
                  baseline=evaluate(arrays['valid_labels'],arrays['baseline_prediction']),
                  incumbent_manifest=current,scope='Small fixed cohort; AUC is comparable only on these same events, not directly to full Jan26 AUC.')
    write_json(out/'metadata.json',metadata)
    write_json(LAB/'generative'/(config['name']+'_data.json'),{'dataset':str(out.relative_to(ROOT)),'config':config})
    print(json.dumps({'dataset':str(out.relative_to(ROOT)),'train_events':len(tr),'valid_events':len(va),
                      'baseline_auc_same_cohort':metadata['baseline']['auc'],'tokens_per_event':32},indent=2))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/generative_smoke.json');a=p.parse_args()
    prepare(json.loads(Path(a.config).read_text()))
