"""Past-only orthogonal alignment for coordinate-aware temporal scoring heads."""
import numpy as np
from arena_core import snapshot
from arena_sgns_features import embedding_model


def orthogonal_map(source,target):
    left,_,right=np.linalg.svd(source.astype(np.float64).T@target.astype(np.float64),full_matrices=False)
    return (left@right).astype(np.float32)


def aligned_features(data,indices,cutoff,config):
    size=config['alignment'].get('factors',8);anchor=config['train_start']
    if anchor>cutoff:raise ValueError('Alignment anchor must precede every feature snapshot')
    options={'epochs':config.get('sgns_epochs',{}).get(str(size)),
             'recipe':config.get('sgns_recipes',{}).get(str(size))}
    source=embedding_model(cutoff,size,**options);stats=np.load(snapshot(cutoff)/'stats.npz')
    if anchor==cutoff:rotation=np.eye(size,dtype=np.float32)
    else:
        reference=embedding_model(anchor,size,**options);prior=np.load(snapshot(anchor)/'stats.npz')
        keep=(stats['pop']>=3)&(prior['pop']>=3)
        if keep.sum()<size:raise ValueError('Insufficient common past items for coordinate alignment')
        a=source['items'][keep];b=reference['items'][keep]
        a=a/np.linalg.norm(a,axis=1).clip(1e-6)[:,None]
        b=b/np.linalg.norm(b,axis=1).clip(1e-6)[:,None]
        rotation=orthogonal_map(a,b)
    users=source['users']@rotation;items=source['items']@rotation
    users[~stats['known_users']]=0;items[stats['pop']==0]=0
    names=[f'aligned{size}_{block}_{j}' for block in ('user','item','history','user_item','history_item') for j in range(size)]
    result=np.empty((len(indices),10,len(names)),dtype=np.float32)
    for start in range(0,len(indices),4096):
        ix=indices[start:start+4096];u=users[data['users'][ix]];v=items[data['items'][ix]]
        h=np.array(data['history'][ix]);valid=(h!=0)&(stats['pop'][h]>0)
        pooled=(items[h]*valid[:,:,None]).sum(1)/valid.sum(1).clip(1)[:,None]
        shape=v.shape;ub=np.broadcast_to(u[:,None,:],shape);hb=np.broadcast_to(pooled[:,None,:],shape)
        result[start:start+len(ix)]=np.concatenate([ub,v,hb,ub*v,hb*v],axis=2)
    return result.reshape(-1,len(names)),names
