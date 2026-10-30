"""Bounded slate reranking trained on chronological first-stage OOF scores."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_two_stage_data import OUT, ROOT, prepare, date_shift
from arena_ood import write_json, guard, sha
from arena_push_model import normalizer
from arena_daily import StackRows

CONFIGS = {
    'residual_mlp': dict(attention=False, gate=False, width=96, scale=.5, rank_weight=0., seed=83801),
    'edge_mlp': dict(attention=False, gate=True, width=96, scale=.5, rank_weight=0., seed=83801),
    'edge_transformer': dict(attention=True, gate=True, width=128, scale=.5, rank_weight=0., seed=83802),
    'edge_transformer_rank': dict(attention=True, gate=True, width=128, scale=.5, rank_weight=.15, seed=83803),
    'origin_edge_mlp_choice': dict(attention=False,gate=True,width=96,scale=.5,rank_weight=0.,seed=83804,
                                   training_mode='origins',origin_days=2,factorized=True,choice_weight=.5),
    'origin_edge_transformer_choice': dict(attention=True,gate=True,width=128,scale=.5,rank_weight=0.,seed=83805,
                                           training_mode='origins',origin_days=2,factorized=True,choice_weight=.5),
    'origin_choice_only': dict(attention=True,gate=True,width=128,scale=.5,rank_weight=0.,seed=83806,
                               training_mode='origins',origin_days=2,factorized=True,choice_weight=1.,freeze_propensity=True),
}


def score_features(prob, pair, history_fraction):
    """Label-free uncertainty and candidate competition, permutation equivariant."""
    p=prob.clamp(1e-7,1-1e-7)
    click=p.sum(1).clamp(1e-6,1-1e-6)
    conditional=p/click[:,None]
    lp=conditional.log()
    top=lp.topk(2,dim=1)
    margin=top.values[:,0]-top.values[:,1]
    entropy=-(conditional*lp).sum(1)/np.log(10)
    tie=torch.exp(-margin/.75)
    sparse=(1-history_fraction.clamp(0,1)).square()
    gate=.15+.85*torch.maximum(tie,.5*sparse)
    event_gate=.25+.75*(4*click*(1-click))
    b,n=prob.shape
    eye=torch.eye(n,device=pair.device,dtype=torch.bool)[None]
    other=pair.masked_fill(eye,0)
    denom=(1-conditional).clamp_min(1e-6)
    expected=(other*conditional[:,None,:]).sum(2)/denom
    rank=(p[:,:,None]>p[:,None,:]).float().mean(2)
    bestcos=pair.gather(2,top.indices[:,0,None,None].expand(-1,n,1)).squeeze(2)
    secondcos=pair.gather(2,top.indices[:,1,None,None].expand(-1,n,1)).squeeze(2)
    maxcos=pair.masked_fill(eye,-2).max(2).values
    gap=(lp[:,:,None]-lp[:,None,:]).clamp(-6,6)
    crowd=(other*torch.exp(-gap.abs())*(~eye)).sum(2)/9
    item=torch.stack([p.log()/4,(lp-top.values[:,0,None])/4,conditional,rank,
                      expected,bestcos,secondcos,maxcos,crowd],2)
    event=torch.stack([click,entropy,margin.clamp(max=8)/4,tie,sparse,event_gate],1)
    return torch.cat([item,event[:,None].expand(-1,n,-1)],2),event,gate,event_gate


class RelationBlock(nn.Module):
    def __init__(self,width,heads=4):
        super().__init__();self.heads=heads;self.width=width
        self.norm1=nn.LayerNorm(width);self.qkv=nn.Linear(width,width*3);self.output=nn.Linear(width,width)
        self.cos_bias=nn.Parameter(torch.zeros(heads))
        self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Dropout(.1),nn.Linear(width*2,width))
        self.drop=nn.Dropout(.1)

    def forward(self,x,pair):
        b,n,w=x.shape;h=self.heads;d=w//h
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,h,d).permute(2,0,3,1,4).unbind(0)
        logits=torch.matmul(q,k.transpose(-1,-2))/np.sqrt(d)+self.cos_bias[None,:,None,None]*pair[:,None]
        msg=torch.matmul(logits.softmax(-1),v).transpose(1,2).reshape(b,n,w)
        x=x+self.drop(self.output(msg));return x+self.drop(self.ff(self.norm2(x)))


class Reranker(nn.Module):
    def __init__(self,mean,scale,config):
        super().__init__();self.config=config;w=config['width'];n=len(mean)
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.embed=nn.Sequential(nn.Linear(n+15,w),nn.SiLU(),nn.Dropout(.1),nn.Linear(w,w),nn.SiLU())
        self.blocks=nn.ModuleList([RelationBlock(w),RelationBlock(w)]) if config['attention'] else nn.ModuleList()
        self.context=nn.Sequential(nn.Linear(w*3+6,96),nn.SiLU(),nn.Dropout(.1),nn.Linear(96,32),nn.SiLU())
        self.rank=nn.Linear(w+32,1);self.propensity=nn.Linear(32,1)
        nn.init.zeros_(self.rank.weight);nn.init.zeros_(self.rank.bias)
        nn.init.zeros_(self.propensity.weight);nn.init.zeros_(self.propensity.bias)
        if config.get('freeze_propensity'):
            for p in self.propensity.parameters():p.requires_grad_(False)

    def forward(self,x,prob,pair):
        p=prob.clamp(1e-7,1-1e-7)
        outside=(1-p.sum(1,keepdim=True)).clamp_min(1e-7)
        base=torch.cat([p,outside],1).log()
        sf,ef,gate,event_gate=score_features(p,pair,x[:,0,25])
        z=((x-self.mean)/self.scale).clamp(-6,6)
        z=self.embed(torch.cat([z,sf],2))
        for block in self.blocks:z=block(z,pair)
        aggregate=torch.cat([z.mean(1),z.max(1).values,z.std(1,unbiased=False),ef],1)
        ctx=self.context(aggregate)
        rank=self.rank(torch.cat([z,ctx[:,None].expand(-1,10,-1)],2)).squeeze(2).tanh()
        rank=rank-rank.mean(1,keepdim=True)
        propensity=self.propensity(ctx).tanh()
        if not self.config['gate']:gate=torch.ones_like(gate);event_gate=torch.ones_like(event_gate)
        if self.config.get('factorized'):
            # Candidate preference and event click probability remain separate.
            # A ranking-only correction preserves the total event click mass.
            choice=F.log_softmax(p.log()+self.config['scale']*rank*gate[:,None],1)
            click_log_odds=p.sum(1,keepdim=True).log()-outside.log()
            click_log_odds=click_log_odds-self.config['scale']*propensity*event_gate[:,None]
            return torch.cat([choice+click_log_odds,torch.zeros_like(outside)],1)
        correction=torch.cat([rank*gate[:,None],propensity*event_gate[:,None]],1)
        return base+self.config['scale']*correction

    def probabilities(self,x,prob,pair):return self(x,prob,pair).softmax(1)[:,:10]


class Inputs:
    def __init__(self,folders):
        self.folders=folders
        self.features=StackRows(*[np.load(f/'features.npy',mmap_mode='r') for f in folders])
        self.pairs=np.concatenate([np.load(f/'pairs.npy') for f in folders])
        scored=[np.load(f/'scores.npz') for f in folders]
        self.prob=np.concatenate([v['predictions'] for v in scored])
        self.labels=np.concatenate([v['labels'] for v in scored])
        self.indices=np.concatenate([v['event_indices'] for v in scored])

    def batch(self,take,device):
        rows=(take[:,None]*10+np.arange(10)).ravel()
        return (torch.tensor(self.features[rows].reshape(-1,10,self.features.shape[1]),device=device),
                torch.tensor(self.prob[take],device=device),torch.tensor(self.pairs[take],device=device))


def training_inputs(cutoff,config):
    if config.get('training_mode')!='origins':
        days=[date_shift(cutoff,-2),date_shift(cutoff,-1)]
        return Inputs([prepare(d) for d in days]),days
    from ctr_data import load_cache
    from run import timestamp
    from arena_daily import EventSubset
    _,raw=load_cache(ROOT/'data/cache_v1','train')
    origins=[date_shift(cutoff,-i) for i in range(config.get('origin_days',2),0,-1)]
    assert min(origins)>='2025-01-22','Earlier first-stage models lack a disjoint warm-up fitting frame.'
    folders=[prepare(d,'valid') for d in origins]
    result=Inputs.__new__(Inputs);result.folders=folders
    features=[];pairs=[];scores=[];labels=[];indices=[]
    for day,f in zip(origins,folders):
        s=np.load(f/'scores.npz');ix=s['event_indices'];pos=np.flatnonzero(raw['timestamps'][ix]<timestamp(cutoff))
        selected=ix[pos]
        # First queries after each origin; all history must precede that origin.
        history=raw['history'][selected];age=raw['age_hours'][selected]
        minutes=raw['timestamps'][selected,None]//60-np.rint(age*60).astype(np.int64)
        assert np.all(minutes[history!=0]<timestamp(day)//60)
        features.append(EventSubset(np.load(f/'features.npy',mmap_mode='r'),pos))
        pairs.append(np.load(f/'pairs.npy',mmap_mode='r')[pos]);scores.append(s['predictions'][pos])
        labels.append(s['labels'][pos]);indices.append(selected)
    result.features=StackRows(*features);result.pairs=np.concatenate(pairs)
    result.prob=np.concatenate(scores);result.labels=np.concatenate(labels);result.indices=np.concatenate(indices)
    return result,origins


def training_objective(z,y,config,prob,pair,history_fraction):
    from arena_objectives import loss as objective
    value=objective(z,y,dict(loss='event_ce',rank_weight=config['rank_weight'],rank_samples=64))
    bp=torch.cat([prob,(1-prob.sum(1,keepdim=True)).clamp_min(1e-7)],1).clamp_min(1e-7)
    value=value+.01*F.kl_div(F.log_softmax(z,1),bp,reduction='batchmean')
    if config.get('choice_weight'):
        _,_,gate,_=score_features(prob,pair,history_fraction)
        clicked=(y.sum(1)>0).float()
        choice=F.cross_entropy(z[:,:10],y.argmax(1),reduction='none')
        value=value+config['choice_weight']*(choice*clicked*gate).mean()
    return value


@torch.inference_mode()
def predict(model,data,device):
    model.eval();parts=[]
    for start in range(0,len(data.indices),2048):
        take=np.arange(start,min(start+2048,len(data.indices)))
        parts.append(model.probabilities(*data.batch(take,device)).cpu().numpy())
    return np.concatenate(parts)


def diagnostics(data,pred):
    from sklearn.metrics import roc_auc_score
    conditional=data.prob/data.prob.sum(1,keepdims=True).clip(1e-8)
    lp=np.log(conditional.clip(1e-8));top=np.sort(lp,1)[:,-2:]
    tie=top[:,1]-top[:,0]<.5
    out={}
    for name,mask in [('near_tie',tie),('confident',~tie)]:
        if np.any(mask) and len(np.unique(data.labels[mask]))==2:
            a=roc_auc_score(data.labels[mask].ravel(),data.prob[mask].ravel())
            b=roc_auc_score(data.labels[mask].ravel(),pred[mask].ravel())
            clicked=data.labels[mask].sum(1)>0
            target=data.labels[mask].argmax(1)
            out[name]=dict(events=int(mask.sum()),stage1_auc=float(a),rerank_auc=float(b),delta=float(b-a),
                clicked_top1_before=float((data.prob[mask].argmax(1)[clicked]==target[clicked]).mean()),
                clicked_top1_after=float((pred[mask].argmax(1)[clicked]==target[clicked]).mean()))
    return out


def train(name,cutoff='2025-01-26',fixed_epochs=None):
    from arena_core import evaluate
    from arena_validation import importance_weights,weighted_auc
    from arena_objectives import loss as objective
    from ctr_data import load_cache
    from run import timestamp
    config=copy.deepcopy(CONFIGS[name]);folder=OUT/'runs'/name/cutoff;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    before=guard()
    data,days=training_inputs(cutoff,config);valid=Inputs([prepare(cutoff,'valid')])
    _,raw=load_cache(ROOT/'data/cache_v1','train')
    assert int(raw['timestamps'][data.indices].max())<timestamp(cutoff)
    assert not np.intersect1d(data.indices,valid.indices).size
    if config.get('training_mode')!='origins':assert len(np.unique(data.indices))==len(data.indices)
    mean,scale=normalizer(data.features,{'normalizer':'relative_range'})
    torch.manual_seed(config['seed']);torch.set_num_threads(2)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=Reranker(mean,scale,config).to(device);ema=copy.deepcopy(model).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.03)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=.00005)
    rng=np.random.default_rng(config['seed']);iw,profile=importance_weights(cutoff,valid.indices)
    record=[];best=-1.;stale=0;epochs=fixed_epochs or 20
    base_metrics=evaluate(valid.labels,valid.prob)
    print(json.dumps(dict(stage='reranker_start',name=name,cutoff=cutoff,training_events=len(data.indices),stage1=base_metrics,
                          parameters=sum(p.numel() for p in model.parameters()))),flush=True)
    for epoch in range(1,epochs+1):
        tick=time.monotonic();order=rng.permutation(len(data.indices));total=0.;model.train()
        for start in range(0,len(order),1024):
            take=order[start:start+1024];x,p,pair=data.batch(take,device)
            y=torch.tensor(data.labels[take],dtype=torch.float32,device=device)
            z=model(x,p,pair)
            loss=training_objective(z,y,config,p,pair,x[:,0,25])
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
            total+=loss.item()*len(take)
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.mul_(.99).add_(b,alpha=.01)
        scheduler.step();prediction=predict(ema,valid,device)
        row=dict(epoch=epoch,train_loss=total/len(order),**evaluate(valid.labels,prediction),weighted_auc=weighted_auc(valid.labels,prediction,iw),seconds=time.monotonic()-tick)
        record.append(row);print(json.dumps(dict(name=name,cutoff=cutoff,**row)),flush=True)
        save=epoch==epochs if fixed_epochs else row['auc']>best
        if save:
            best=row['auc'];stale=0
            torch.save(dict(state_dict=ema.state_dict(),config=config,width=len(mean)),folder/'model.pt')
            np.savez_compressed(folder/'validation.npz',event_indices=valid.indices,labels=valid.labels,predictions=prediction)
        else:stale+=1
        write_json(folder/'training.json',record)
        if not fixed_epochs and stale>=4:break
    winner=record[-1] if fixed_epochs else max(record,key=lambda r:r['auc'])
    stored=np.load(folder/'validation.npz')['predictions']
    result=dict(name=name,cutoff=cutoff,config=config,best_epoch=winner['epoch'],**winner,
                stage1=base_metrics,cohorts=diagnostics(valid,stored),days=days,training_events=len(data.indices),
                protocol='Daily chronological OOF fine ranker, then bounded residual slate reranker. Secondary/third origins use primary-locked epochs.',
                training_inputs=[json.loads((f/'protocol.json').read_text()) for f in data.folders],cohort_profile=profile,
                parameters=sum(p.numel() for p in model.parameters()),unique_training_events=len(np.unique(data.indices)),
                maximum_training_target_timestamp=int(raw['timestamps'][data.indices].max()))
    write_json(folder/'result.json',result)
    assert guard()==before
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=list(CONFIGS));p.add_argument('--cutoff',default='2025-01-26')
    p.add_argument('--fixed-epochs',type=int);a=p.parse_args();train(a.name,a.cutoff,a.fixed_epochs)
