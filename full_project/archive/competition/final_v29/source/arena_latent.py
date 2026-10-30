"""Time-safe latent factors and normalized matching features for the CTR head."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/".tools/python"))
import json
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from implicit.cpu.als import AlternatingLeastSquares
from ctr_data import load_cache
from arena_core import snapshot
from arena_report import note


def latent_model(cutoff,factors=32,alpha=10,regularization=0.1,iterations=25):
    parent=snapshot(cutoff)
    folder=parent/f"als_{factors}_{alpha}_{regularization}_{iterations}"
    model=AlternatingLeastSquares(factors=factors,alpha=alpha,regularization=regularization,
                                 iterations=iterations,num_threads=6,random_state=42)
    if (folder/"factors.npz").exists():
        values=np.load(folder/"factors.npz");model.user_factors=values["users"];model.item_factors=values["items"]
        return model
    folder.mkdir(exist_ok=True)
    with threadpool_limits(limits=1,user_api="blas"):
        model.fit(sparse.load_npz(parent/"interactions.npz"),show_progress=False)
    np.savez(folder/"factors.npz",users=model.user_factors,items=model.item_factors)
    (folder/"config.json").write_text(json.dumps({"cutoff":cutoff,"factors":factors,"alpha":alpha,
        "regularization":regularization,"iterations":iterations}))
    print(f"latent snapshot {cutoff} factors={factors} ready",flush=True)
    return model


def latent_features(data,indices,cutoff,factors=32,alpha=10,regularization=0.1):
    model=latent_model(cutoff,factors,alpha,regularization)
    n=len(indices)
    items=np.array(data["items"][indices]);users=np.array(data["users"][indices])
    history=np.array(data["history"][indices]);ages=np.array(data["age_hours"][indices])
    keep=history.ravel()!=0
    q=sparse.coo_matrix((np.ones(keep.sum(),dtype=np.float32),
        (np.repeat(np.arange(n),20)[keep],history.ravel()[keep])),shape=(n,len(model.item_factors))).tocsr()
    q.data=1+np.log(q.data)
    with threadpool_limits(limits=1,user_api="blas"):
        folded=model.recalculate_user(np.arange(n),q)
    known=model.user_factors[users].copy()
    unknown=(known*known).sum(1)<1e-12
    known[unknown]=folded[unknown]
    names=[f"als{factors}_{s}" for s in ("known_dot","fold_dot","known_cos","fold_cos","known_norm","fold_norm",
            "item_norm","history_dot_mean","history_cos_mean","history_cos_max","history_cos_recent",
            "known_relative","fold_relative","cos_relative","fold_cos_relative")]
    out=np.zeros((n,10,len(names)),dtype=np.float32)
    for start in range(0,n,4096):
        end=min(n,start+4096)
        v=model.item_factors[items[start:end]]
        u=known[start:end];f=folded[start:end]
        un=np.linalg.norm(u,axis=1).clip(1e-6);fn=np.linalg.norm(f,axis=1).clip(1e-6)
        vn=np.linalg.norm(v,axis=2).clip(1e-6)
        ud=(v*u[:,None]).sum(2);fd=(v*f[:,None]).sum(2)
        uc=ud/(un[:,None]*vn);fc=fd/(fn[:,None]*vn)
        h=model.item_factors[history[start:end]]
        hn=np.linalg.norm(h,axis=2).clip(1e-6)
        valid=history[start:end]!=0
        hd=np.einsum("bik,bjk->bij",v,h)
        hc=hd/(vn[:,:,None]*hn[:,None,:]);hc*=valid[:,None,:]
        length=valid.sum(1).clip(1)
        recency=np.exp(-ages[start:end]/72)*valid
        recent=(hc*recency[:,None,:]).sum(2)/recency.sum(1).clip(1e-6)[:,None]
        arrays=[ud,fd,uc,fc,np.broadcast_to(un[:,None],ud.shape),np.broadcast_to(fn[:,None],ud.shape),vn,
                (hd*valid[:,None,:]).sum(2)/length[:,None],hc.sum(2)/length[:,None],hc.max(2),recent,
                ud-ud.mean(1,keepdims=True),fd-fd.mean(1,keepdims=True),uc-uc.mean(1,keepdims=True),fc-fc.mean(1,keepdims=True)]
        out[start:end]=np.stack(arrays,axis=2)
    return out.reshape(-1,len(names)),names
