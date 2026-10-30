"""Longer spherical representation training with fixed, past-only checkpoints.

The accepted 32-epoch embeddings improved through their final checkpoint.
This independent recipe checks 32/64/128 under one fixed long LR schedule;
query labels never enter the representation builder or checkpoint selection.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import copy
import fcntl
import json
import time
import numpy as np
import torch
from scipy import sparse
from arena_ood import ROOT,write_json,sha,guard
from arena_spherical_probe import Spherical

RECIPE=dict(version=1,dimension=8,epochs=128,outputs=[32,64,128],seed=82719,lr=.002,
            minimum_lr=.0002,weight_decay=.001,ema=.99,negatives=8,batch_size=8192,
            anchor=.02,initial_temperature=6.,device='cpu',schedule='cosine128')
TAG='spherical_long_v1_d8_e128'


def model(cutoff):
    from arena_core import snapshot
    from arena_sgns_features import embedding_model
    from run import timestamp
    folder=snapshot(cutoff)/TAG;folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            assert json.loads((folder/'result.json').read_text())['recipe']==RECIPE
            return folder
        metadata=json.loads((snapshot(cutoff)/'metadata.json').read_text())
        assert metadata['max_fit_timestamp']<timestamp(cutoff)
        initial=embedding_model(cutoff,8,recipe=dict(epochs=320,lr=.01,lr_decay=False,seed=2026,
                                reuse_experiment='sgns8_const_e320_s2026'))
        x=sparse.load_npz(snapshot(cutoff)/'interactions.npz')
        ku,ki=np.diff(x.indptr)>0,np.asarray(x.sum(0)).ravel()>0;ku[0]=ki[0]=False
        pu,pi=x.nonzero();pu,pi=torch.tensor(pu,dtype=torch.long),torch.tensor(pi,dtype=torch.long)
        torch.manual_seed(RECIPE['seed']);torch.set_num_threads(2)
        net=Spherical(initial,ku,ki);ema=copy.deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=128,eta_min=.0002)
        records=[];start_time=time.monotonic();start_epoch=1
        if (folder/'resume.pt').exists():
            checkpoint=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
            assert checkpoint['recipe']==RECIPE
            net.load_state_dict(checkpoint['net']);ema.load_state_dict(checkpoint['ema'])
            opt.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
            torch.random.set_rng_state(checkpoint['rng']);records=checkpoint['records'];start_epoch=checkpoint['epoch']+1
        for epoch in range(start_epoch,129):
            tick=time.monotonic();order=torch.randperm(len(pu));total=0.
            for start in range(0,len(pu),8192):
                chosen=order[start:start+8192]
                negative=torch.randint(1,len(initial['items']),(len(chosen),8))
                loss=net.loss(pu[chosen],pi[chosen],negative)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
                with torch.no_grad():
                    for target,p in zip(ema.parameters(),net.parameters()):target.mul_(.99).add_(p,alpha=.01)
                total+=loss.item()*len(chosen)
            scheduler.step();row=dict(epoch=epoch,loss=total/len(pu),seconds=time.monotonic()-tick)
            records.append(row)
            if epoch in RECIPE['outputs']:
                np.savez(folder/('factors_e'+str(epoch)+'.npz'),**ema.representation())
            if epoch%8==0:
                torch.save(dict(recipe=RECIPE,epoch=epoch,net=net.state_dict(),ema=ema.state_dict(),
                                optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=torch.random.get_rng_state(),records=records),folder/'resume.tmp.pt')
                os.replace(folder/'resume.tmp.pt',folder/'resume.pt')
                write_json(folder/'training.json',records)
                print(json.dumps(dict(stage='spherical_long',cutoff=cutoff,**row)),flush=True)
        write_json(folder/'result.json',dict(cutoff=cutoff,recipe=RECIPE,max_fit_timestamp=metadata['max_fit_timestamp'],
                   snapshot_metadata=metadata,positive_pairs=int(x.nnz),seconds=time.monotonic()-start_time,
                   model_sha256={str(e):sha(folder/('factors_e'+str(e)+'.npz')) for e in RECIPE['outputs']},
                   policy='All representation fitting clicks strictly precede cutoff; fixed checkpoints and LR schedule; query labels never accessed.'))
    return folder


def probe(day):
    from ctr_data import load_cache,temporal_split
    from run import timestamp
    from sklearn.metrics import roc_auc_score
    cutoff='2025-01-'+day;out=ROOT/'competition/spherical_long_probe'/day;out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():return
    before=guard();folder=model(cutoff)
    _,raw=load_cache(ROOT/'data/cache_v1','train');_,valid=temporal_split(raw,timestamp(cutoff))
    records=[]
    for epoch in RECIPE['outputs']:
        rep=dict(np.load(folder/('factors_e'+str(epoch)+'.npz')))
        score=(rep['users'][raw['users'][valid],None]*rep['items'][raw['items'][valid]]).sum(2)
        records.append(dict(epoch=epoch,cosine_auc=float(roc_auc_score(raw['labels'][valid].ravel(),score.ravel()))))
    write_json(out/'report.json',dict(cutoff=cutoff,recipe=RECIPE,records=records,
               source=str(folder.relative_to(ROOT)),policy='Diagnostic only; all three checkpoints fixed before query metrics, no CSV change.'))
    assert all(sha(ROOT/p)==d for p,d in before.items() if p!='competition/CURRENT_BEST.json')
    print(json.dumps(dict(stage='spherical_long_probe_completed',day=day,records=records)),flush=True)


def extra_features(data,indices,cutoff,config=None):
    from arena_coclick import describe
    from run import timestamp
    assert np.all(data['timestamps'][indices]>=timestamp(cutoff))
    folder=model(cutoff);parts=[];names=[]
    for epoch in [64,128]:
        rep=dict(np.load(folder/('factors_e'+str(epoch)+'.npz')))
        x,cols=describe(rep,data,indices);parts.append(x)
        names += [c.replace('coclick_','spherelong'+str(epoch)+'_',1) for c in cols]
    return np.concatenate(parts,1),names


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--day',required=True,choices=['26','25']);a=p.parse_args();probe(a.day)
