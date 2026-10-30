"""Fixed-epoch auxiliary embeddings, fit before each CTR feature window."""
import fcntl
import json
import shutil
import hashlib
import numpy as np
from arena_core import snapshot


def embedding_model(cutoff,factors=16,seed=42,epochs=None,recipe=None):
    if epochs is None:epochs=80 if factors<=8 else 56
    if recipe:epochs=recipe.get("epochs",epochs);seed=recipe.get("seed",seed)
    tag=f"sgns_{factors}_s{seed}_e{epochs}_t80"
    if recipe:tag+="_r"+hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()[:10]
    folder=snapshot(cutoff)/tag
    folder.mkdir(exist_ok=True)
    with (folder/".build.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (folder/"training_result.json").exists():
            source_name=recipe.get("reuse_experiment") if recipe else None
            source=__import__('arena_report').LAB/"experiments"/(source_name or f"sgns{factors}_n8_e80_s{seed}")
            if cutoff=="2025-01-26" and (source/"result.json").exists():
                result=json.loads((source/"result.json").read_text())
                if result["best_epoch"]==epochs:
                    shutil.copy2(source/"factors.npz",folder/"factors.npz")
                    (folder/"training_result.json").write_text(json.dumps({"reused_from":str(source),"epochs":epochs}))
                    return dict(np.load(folder/"factors.npz"))
            from arena_sgns import run
            config={"name":folder.name,"output_dir":str(folder),"factors":factors,"seed":seed,"epochs":epochs,
                    "cutoff":cutoff,"device":"mps","negatives":8,"lr":0.02,"batch_size":8192,
                    "regularization":0.001,"lr_decay":True,"lr_schedule_epochs":80,"evaluate":False}
            if recipe:config.update({key:value for key,value in recipe.items() if key!="reuse_experiment"})
            run(config)
    return dict(np.load(folder/"factors.npz"))


def embedding_features(data,indices,cutoff,factors=16,seed=42,epochs=None,recipe=None):
    if recipe:seed=recipe.get('seed',seed)
    model=embedding_model(cutoff,factors,seed,epochs,recipe)
    users=np.array(data["users"][indices]);items=np.array(data["items"][indices]);history=np.array(data["history"][indices])
    ages=np.array(data["age_hours"][indices]);n=len(indices)
    names=[f"sgns{factors}_s{seed}_{s}" for s in ("dot","cos","usernorm","itemnorm","item_bias","hist_cos_mean",
        "hist_cos_max","hist_cos_recent","dot_relative","cos_relative","hist_same_bucket_cos","hist_same_weekend_cos")]
    out=np.empty((n,10,len(names)),dtype=np.float32)
    for start in range(0,n,4096):
        end=min(n,start+4096);u=model["users"][users[start:end]];v=model["items"][items[start:end]]
        un=np.linalg.norm(u,axis=1).clip(1e-6);vn=np.linalg.norm(v,axis=2).clip(1e-6)
        dot=(v*u[:,None,:]).sum(2);cos=dot/(un[:,None]*vn)
        h=model["items"][history[start:end]];hn=np.linalg.norm(h,axis=2).clip(1e-6)
        valid=history[start:end]!=0;hc=np.einsum('bik,bjk->bij',v,h)/(vn[:,:,None]*hn[:,None,:]);hc*=valid[:,None,:]
        length=valid.sum(1).clip(1);w=np.exp(-ages[start:end]/72)*valid
        ts=np.array(data["timestamps"][indices[start:end]])
        past=ts[:,None]//60-np.rint(ages[start:end]*60).astype(np.int64)
        same_bucket=((past%1440)//360==(ts%86400//21600)[:,None])&valid
        same_day=((((past//1440+3)%7)>=5)==(np.array(data["context"][indices[start:end],2])>0)[:,None])&valid
        arrays=[dot,cos,np.broadcast_to(un[:,None],dot.shape),vn,model["bias"][items[start:end]],
                hc.sum(2)/length[:,None],hc.max(2),(hc*w[:,None,:]).sum(2)/w.sum(1).clip(1e-6)[:,None],
                dot-dot.mean(1,keepdims=True),cos-cos.mean(1,keepdims=True),
                (hc*same_bucket[:,None,:]).sum(2)/same_bucket.sum(1).clip(1)[:,None],
                (hc*same_day[:,None,:]).sum(2)/same_day.sum(1).clip(1)[:,None]]
        out[start:end]=np.stack(arrays,axis=2)
    return out.reshape(-1,len(names)),names
