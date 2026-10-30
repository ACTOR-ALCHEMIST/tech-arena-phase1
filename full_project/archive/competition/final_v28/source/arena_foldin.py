"""Infer a query user's profile from prior clicks with item factors held fixed."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy import sparse
from arena_core import snapshot
from arena_sgns_features import embedding_model


def optimize_profiles(u0,item_vectors,item_bias,history,ages,steps=30,lr=0.1,anchor=0.01,negatives=512,device='mps'):
    """No current-event labels. Sum over users keeps updates batch independent."""
    device=torch.device(device)
    v=torch.tensor(item_vectors,device=device);bias=torch.tensor(item_bias,device=device)
    h=torch.tensor(history,dtype=torch.long,device=device);age=torch.tensor(ages,device=device)
    baseline=torch.tensor(u0,device=device);u=torch.nn.Parameter(baseline.clone())
    hv=v[h];hb=bias[h];valid=h!=0
    weight=torch.exp(-age/168)*valid;weight/=weight.sum(1,keepdim=True).clamp_min(1e-6)
    ids=np.random.default_rng(12345).integers(1,len(item_vectors),size=negatives)
    nv=v[torch.tensor(ids,device=device)];nb=bias[torch.tensor(ids,device=device)]
    optimizer=torch.optim.Adam([u],lr=lr)
    active=valid.any(1)
    for _ in range(steps):
        pos=(u[:,None,:]*hv).sum(2)+hb-np.log(8)
        neg=u@nv.T+nb-np.log(8)
        per_user=(F.softplus(-pos)*weight).sum(1)+8*F.softplus(neg).mean(1)+anchor*(u-baseline).square().mean(1)
        loss=(per_user*active).sum()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    return u.detach().cpu().numpy()


def foldin_features(data,indices,cutoff,config):
    options=config.get('foldin',{});factors=options.get('factors',8)
    model=embedding_model(cutoff,factors,epochs=config.get('sgns_epochs',{}).get(str(factors)),recipe=config.get('sgns_recipes',{}).get(str(factors)))
    path=snapshot(cutoff);x=sparse.load_npz(path/'interactions.npz');stats=np.load(path/'stats.npz')
    known_users=np.diff(x.indptr)>0;known_items=stats['pop']>0
    names=[f'sgns{factors}_foldin_{s}' for s in ['dot','cos','norm','dot_relative','cos_relative','change']]
    result=np.empty((len(indices),10,len(names)),np.float32)
    for start in range(0,len(indices),4096):
        ix=indices[start:start+4096];users=np.array(data['users'][ix]);items=np.array(data['items'][ix])
        history=np.array(data['history'][ix]);history[~known_items[history]]=0
        u0=model['users'][users].copy();u0[~known_users[users]]=0
        fitted=optimize_profiles(u0,model['items'],model['bias'],history,np.array(data['age_hours'][ix]),
                    steps=options.get('steps',30),lr=options.get('lr',0.1),anchor=options.get('anchor',0.01),negatives=options.get('negatives',512))
        v=model['items'][items].copy();v[~known_items[items]]=0
        norm=np.linalg.norm(fitted,axis=1).clip(1e-6);vn=np.linalg.norm(v,axis=2).clip(1e-6)
        dot=(v*fitted[:,None,:]).sum(2);cos=dot/(norm[:,None]*vn)
        change=np.linalg.norm(fitted-u0,axis=1)
        result[start:start+len(ix)]=np.stack([dot,cos,np.broadcast_to(norm[:,None],dot.shape),
                    dot-dot.mean(1,keepdims=True),cos-cos.mean(1,keepdims=True),np.broadcast_to(change[:,None],dot.shape)],2)
    return result.reshape(-1,len(names)),names
