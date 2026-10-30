"""Chronological target-aware history reranking, evaluated on frozen incumbent."""
import argparse
import copy
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from arena_ood import ROOT,write_json,guard
from arena_history_data import OUT,Data
from arena_history_model import HistoryReranker,CONFIGS
from arena_push_model import normalizer
from arena_metric_geometry import preference_probabilities


@torch.inference_mode()
def predict_delta(net,data,device):
    net.eval();parts=[]
    for start in range(0,len(data.indices),512):
        take=np.arange(start,min(start+512,len(data.indices)))
        parts.append(net.corrections(*data.batch(take,device)).cpu().numpy())
    return np.concatenate(parts)


def metrics(labels,p,base):
    from sklearn.metrics import roc_auc_score
    lp=np.sort(np.log(base.clip(1e-8)),axis=1);near=lp[:,-1]-lp[:,-2]<.5
    clicked=labels.sum(1)>0;mask=near&clicked
    return dict(auc=float(roc_auc_score(labels.ravel(),p.ravel())),
                near_tie_top1=float((p[mask].argmax(1)==labels[mask].argmax(1)).mean()),
                all_clicked_top1=float((p[clicked].argmax(1)==labels[clicked].argmax(1)).mean()))


def train(name,cutoff='2025-01-26',fixed_epochs=None,smoke=False):
    from ctr_data import load_cache
    from arena_validation import importance_weights,weighted_auc
    from arena_objectives import loss as objective
    config=copy.deepcopy(CONFIGS[name]);folder=OUT/('smoke' if smoke else 'runs')/name/cutoff;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    before=guard()
    if not (OUT/'incumbent_pointer.json').exists():write_json(OUT/'incumbent_pointer.json',json.loads((ROOT/'competition/CURRENT_BEST.json').read_text()))
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text());parent=(ROOT/pointer['csv']).parent
    _,raw=load_cache(ROOT/'data/cache_v1','train');data=Data(cutoff,raw,True);valid=Data(cutoff,raw)
    old=np.load(parent/('validation_'+cutoff[-2:]+'.npz'))
    for k,a in [('labels',valid.labels),('event_indices',valid.indices)]:np.testing.assert_array_equal(old[k],a)
    assert not np.intersect1d(data.indices,valid.indices).size
    mean,scale=normalizer(data.features,dict(normalizer='relative_range'))
    torch.manual_seed(config['seed']);torch.set_num_threads(2);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net=HistoryReranker(mean,scale,config).to(device);ema=copy.deepcopy(net).eval()
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.03)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=.00005)
    iw,_=importance_weights(cutoff,valid.indices);rng=np.random.default_rng(config['seed'])
    active=np.arange(len(data.indices))
    if smoke:active=np.sort(rng.choice(active,min(5000,len(active)),replace=False))
    records=[];best=-1.;stale=0;epochs=2 if smoke else (fixed_epochs if fixed_epochs is not None else 16)
    for epoch in range(epochs+1):
        tick=time.monotonic();total=0.
        if epoch:
            net.train();order=rng.permutation(active)
            for start in range(0,len(order),256):
                take=order[start:start+256];inputs=data.batch(take,device);p=inputs[1]
                y=torch.tensor(data.labels[take],dtype=torch.float32,device=device);z=net(*inputs)
                loss=objective(z,y,dict(loss='event_ce',rank_weight=config['rank_weight'],rank_samples=64))
                qnew=F.softmax(z[:,:10],1);qbase=p/p.sum(1,keepdim=True).clamp_min(1e-8)
                loss=loss+.02*(qnew*(qnew.clamp_min(1e-8).log()-qbase.clamp_min(1e-8).log())).sum(1).mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
                total+=loss.item()*len(take)
                with torch.no_grad():
                    for a,b in zip(ema.parameters(),net.parameters()):a.mul_(.99).add_(b,alpha=.01)
            sched.step()
        delta=predict_delta(ema,valid,device)
        p=preference_probabilities(old['predictions'],delta,1.)
        row=dict(epoch=epoch,train_loss=total/len(active) if epoch else None,**metrics(valid.labels,p,old['predictions']),
                 weighted_auc=weighted_auc(valid.labels,p,iw),seconds=time.monotonic()-tick)
        records.append(row);print(json.dumps(dict(name=name,cutoff=cutoff,smoke=smoke,**row)),flush=True)
        save=epoch==epochs if fixed_epochs is not None else row['auc']>best
        if save:
            best=row['auc'];stale=0
            torch.save(dict(state_dict=ema.state_dict(),config=config,width=len(mean)),folder/'model.pt')
            np.savez_compressed(folder/'validation.npz',event_indices=valid.indices,labels=valid.labels,predictions=p,correction=delta)
        else:stale+=1
        write_json(folder/'training.json',records)
        if fixed_epochs is None and not smoke and stale>=4:break
    winner=records[-1] if fixed_epochs is not None else max(records,key=lambda r:r['auc'])
    result=dict(name=name,cutoff=cutoff,config=config,**winner,best_epoch=winner['epoch'],training_events=len(active),
                parameters=sum(p.numel() for p in net.parameters()),smoke=smoke,
                policy='Strong time-OOF fine scores; target/history similarities have prior-only frames. Residual transferred onto frozen incumbent, click mass preserved.',
                frames=np.unique(data.frames).tolist(),baseline=pointer)
    write_json(folder/'result.json',result)
    assert guard()==before
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=list(CONFIGS));p.add_argument('--cutoff',default='2025-01-26')
    p.add_argument('--fixed-epochs',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args();train(a.name,a.cutoff,a.fixed_epochs,a.smoke)
