"""Chronological, out-of-sample inputs for a second-stage slate reranker.

An OOF day is predicted by a ranker fitted strictly before midnight that day.
Target labels are stored separately and never passed to feature construction.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.55')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.4')
import argparse
import copy
import json
import time
from datetime import datetime, timezone
import numpy as np
import torch
from arena_ood import ROOT, sha, write_json, guard
from arena_push_model import build_model, normalizer
from arena_tabular import predict

OUT = ROOT/'competition/two_stage'
STAGE1_NAME = 'spectral_multi_wide'
FIXED_STAGE1_EPOCHS = 10


def date_shift(day, offset):
    from run import timestamp
    return datetime.fromtimestamp(timestamp(day)+offset*86400, timezone.utc).strftime('%Y-%m-%d')


def day_indices(data, day):
    from run import timestamp
    t = np.asarray(data['timestamps'])
    return np.flatnonzero((t >= timestamp(day)) & (t < timestamp(day)+86400))


def raw_inputs(data):
    return {k:v for k,v in data.items() if k != 'labels'}


def ranker_path(day):
    old = ROOT/'competition/ood/runs'/STAGE1_NAME/day
    return old if (old/'model.pt').exists() else OUT/'stage1'/day


def ranker_recipe(day):
    from arena_ood import source
    ref = json.loads((ROOT/'competition/ood/runs'/STAGE1_NAME/'2025-01-26/result.json').read_text())
    fc = copy.deepcopy(ref['feature_config'])
    fc['train_start'] = max('2025-01-21', date_shift(day, -2))
    cfg = copy.deepcopy(ref['config'])
    return cfg, fc


def stage1_features(inputs, indices, frame, fc):
    from arena_features import complete_features
    from arena_ood_features import extension_features
    a, an = complete_features(inputs, indices, frame, fc)
    b, bn = extension_features(inputs, indices, frame, fc, 'spectral_multi')
    return np.concatenate([a,b], 1), an+bn


def fit_ranker(day):
    """Earlier folds use ten fixed epochs, with no held-out-day access."""
    from ctr_data import load_cache
    from run import timestamp
    folder = ranker_path(day)
    if (folder/'model.pt').exists() and (folder/'result.json').exists():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    _, data = load_cache(ROOT/'data/cache_v1', 'train')
    cfg, fc = ranker_recipe(day)
    ix = np.flatnonzero((data['timestamps'] >= timestamp(fc['train_start'])) &
                        (data['timestamps'] < timestamp(day)))
    if len(ix) > 180000:
        ix = np.sort(np.random.default_rng(42).choice(ix, 180000, replace=False))
    assert len(ix) and int(data['timestamps'][ix].max()) < timestamp(day)
    np.save(folder/'training_indices.npy', ix)
    print(json.dumps(dict(stage='earlier_ranker_features', day=day, events=len(ix))), flush=True)
    cache = folder/'working_features.npy'
    if not cache.exists():
        x, names = stage1_features(raw_inputs(data), ix, fc['train_start'], fc)
        np.save(cache, x)
        write_json(folder/'features.json', dict(names=names, frame=fc['train_start'], indices_sha256=sha(folder/'training_indices.npy')))
        del x
    x = np.load(cache, mmap_mode='r')
    mean, scale = normalizer(x, cfg)
    torch.manual_seed(cfg['seed']); torch.set_num_threads(2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = build_model(x.shape[1], mean, scale, cfg).to(device)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=.0001)
    rng = np.random.default_rng(cfg['seed'])
    labels = data['labels'][ix]
    target = np.where(labels.sum(1)>0, labels.argmax(1), 10)
    records = []
    for epoch in range(1, FIXED_STAGE1_EPOCHS+1):
        tick=time.monotonic();model.train();order=rng.permutation(len(ix));total=0.
        for start in range(0,len(order),1024):
            take=order[start:start+1024];rows=(take[:,None]*10+np.arange(10)).ravel()
            bx=torch.tensor(np.array(x[rows]).reshape(-1,10,x.shape[1]),device=device)
            y=torch.tensor(target[take],device=device)
            loss=torch.nn.functional.cross_entropy(model(bx),y)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            total+=loss.item()*len(take)
            with torch.no_grad():
                for p,q in zip(ema.parameters(),model.parameters()):p.mul_(.995).add_(q,alpha=.005)
        schedule.step()
        records.append(dict(epoch=epoch,loss=total/len(ix),seconds=time.monotonic()-tick))
        print(json.dumps(dict(stage='earlier_ranker_fit',day=day,**records[-1])),flush=True)
        write_json(folder/'training.json',records)
    torch.save(dict(state_dict=ema.cpu().state_dict(),width=x.shape[1],loss='event_ce',config=cfg),folder/'model.pt')
    write_json(folder/'result.json',dict(day=day,config=cfg,feature_config=fc,best_epoch=FIXED_STAGE1_EPOCHS,
               training_events=len(ix),maximum_fit_timestamp=int(data['timestamps'][ix].max()),
               policy='Fixed ten epochs inherited from existing ranker; no held-out-day labels or metrics used.'))
    del x
    write_json(folder/'working_cleanup.json',dict(path=str(cache.relative_to(ROOT)),sha256=sha(cache),bytes=cache.stat().st_size,
               policy='Regenerable training feature matrix only; model, indices, recipe and raw input preserved.'))
    cache.unlink()
    return folder


def load_ranker(day, device):
    folder=fit_ranker(day)
    c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    state=c['state_dict']; net=build_model(c['width'],state['mean'].numpy(),state['scale'].numpy(),c['config'])
    net.load_state_dict(state);return net.to(device).eval(), folder


def ranker_fit_provenance(folder,day,data):
    from run import timestamp
    result=json.loads((folder/'result.json').read_text())
    if (folder/'training_indices.npy').exists():
        ix=np.load(folder/'training_indices.npy')
    else:
        ix=np.load(ROOT/result['feature_dataset']/'labels.npz')['train_indices']
    maximum=int(data['timestamps'][ix].max())
    assert maximum<timestamp(day)
    return dict(training_events=len(ix),minimum_fit_timestamp=int(data['timestamps'][ix].min()),
                maximum_fit_timestamp=maximum,feature_fitting_frame=result['feature_config']['train_start'],
                query_target_boundary=timestamp(day),ranker_sha256=sha(folder/'model.pt'))


def relation_features(rep, inputs, ix):
    """Only rotation-invariant item relations; no categorical numeric values."""
    v=rep['items'][inputs['items'][ix]].astype(np.float32)
    v/=np.linalg.norm(v,axis=2,keepdims=True).clip(1e-8)
    return np.einsum('bid,bjd->bij',v,v).astype(np.float32)


def prepare(day, kind='oof'):
    from ctr_data import load_cache, temporal_split
    from run import timestamp
    from arena_spherical_repr import model as sphere_model, extra_features
    before=guard()
    folder=OUT/'inputs'/(kind+'_'+day);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'protocol.json').exists() and all((folder/p).exists() for p in ['features.npy','scores.npz','pairs.npy']):
        return folder
    _,data=load_cache(ROOT/'data/cache_v1','train')
    ix=day_indices(data,day) if kind=='oof' else temporal_split(data,timestamp(day))[1]
    inputs=raw_inputs(data)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');torch.set_num_threads(2)
    model,rankfolder=load_ranker(day,device)
    cfg,fc=ranker_recipe(day)
    if not (folder/'features.npy').exists():
        x,names=stage1_features(inputs,ix,day,fc)
        predictions=predict(model,x,device)
        sphere,cols=extra_features(inputs,ix,day,fc)
        mm=np.lib.format.open_memmap(folder/'features.tmp.npy',mode='w+',dtype=np.float32,shape=(len(ix)*10,x.shape[1]+sphere.shape[1]))
        mm[:,:x.shape[1]]=x;mm[:,x.shape[1]:]=sphere;mm.flush();del mm
        os.replace(folder/'features.tmp.npy',folder/'features.npy')
        write_json(folder/'columns.json',names+cols)
        del x,sphere
    else:
        # Interrupted feature generation can resume without using query labels.
        x=np.load(folder/'features.npy',mmap_mode='r')
        from arena_daily import EventSubset
        class First200:
            shape=(len(x),200)
            def __len__(self):return len(x)
            def __getitem__(self,k):return x[k,:200]
        predictions=predict(model,First200(),device)
    rep=dict(np.load(sphere_model(day)/'factors_e32.npz'))
    pairs=relation_features(rep,inputs,ix);np.save(folder/'pairs.npy',pairs)
    np.savez_compressed(folder/'scores.npz',event_indices=ix,predictions=predictions,labels=data['labels'][ix])
    assert int(data['timestamps'][ix].min()) >= timestamp(day)
    from arena_core import snapshot
    sm=json.loads((snapshot(day)/'metadata.json').read_text())
    assert sm['max_fit_timestamp'] < timestamp(day)
    provenance=ranker_fit_provenance(rankfolder,day,data)
    write_json(folder/'protocol.json',dict(day=day,kind=kind,events=len(ix),features_cutoff=day,
               min_query_timestamp=int(data['timestamps'][ix].min()),max_query_timestamp=int(data['timestamps'][ix].max()),
               max_snapshot_fit_timestamp=sm['max_fit_timestamp'],ranker=str(rankfolder.relative_to(ROOT)),
               ranker_sha256=sha(rankfolder/'model.pt'),ranker_fitting_provenance=provenance,feature_columns=244,
               label_policy='Query labels excluded from feature functions and first-stage fitting. Scores truly chronological OOF. Full 10-item slates retained.'))
    assert all(sha(ROOT/p)==s for p,s in before.items())
    print(json.dumps(dict(stage='two_stage_inputs_ready',day=day,kind=kind,events=len(ix))),flush=True)
    return folder


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['ranker','prepare']);p.add_argument('--day',required=True)
    p.add_argument('--kind',default='oof',choices=['oof','valid']);a=p.parse_args()
    fit_ranker(a.day) if a.command=='ranker' else prepare(a.day,a.kind)
