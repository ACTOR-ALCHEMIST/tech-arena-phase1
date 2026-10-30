"""NumPy-only inference geometry for the small OOF metric reranker."""
import numpy as np


def profiles(raw,ix,rep):
    u=rep['users'][raw['users'][ix]].astype(np.float32)
    items=rep['items'].astype(np.float32)
    items=items/np.linalg.norm(items,axis=1,keepdims=True).clip(1e-8)
    h=items[raw['history'][ix]];visible=raw['history'][ix]!=0
    rank=np.cumsum(visible,axis=1);length=visible.sum(1)
    recent=visible & (rank>length[:,None]-3)
    last=(h*recent[:,:,None]).sum(1)/recent.sum(1).clip(1)[:,None]
    w=np.exp(-raw['age_hours'][ix]/72)*visible
    decay=(h*w[:,:,None]).sum(1)/w.sum(1).clip(1e-6)[:,None]
    u=u/np.linalg.norm(u,axis=1,keepdims=True).clip(1e-8)
    return np.stack([u,last,decay],1).astype(np.float32),items[raw['items'][ix]].astype(np.float32)


def correction(profile,candidates,matrix):
    # Match the two explicit contractions used by the training model.
    q=np.einsum('bkd,kde->be',profile[:,:len(matrix)],matrix)
    return np.einsum('be,bie->bi',q,candidates)


def preference_probabilities(base,delta,mass):
    p=np.asarray(base,dtype=np.float64).clip(1e-8,1-1e-8)
    z=np.log(p)+mass*np.asarray(delta,dtype=np.float64)
    z-=z.max(1,keepdims=True);q=np.exp(z);q/=q.sum(1,keepdims=True)
    return q*p.sum(1,keepdims=True)
