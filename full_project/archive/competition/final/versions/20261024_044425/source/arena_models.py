"""Model families evaluated against the same complete, future event cohort."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "6")
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/".tools/python"))

import argparse
import json
import hashlib
import time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from ctr_data import load_cache, temporal_split
from run import timestamp
from arena_core import snapshot, features, evaluate, sigmoid
from arena_report import LAB, note, write_json


def save_result(out, result, valid, labels, predictions):
    np.savez_compressed(out/"validation.npz", event_indices=valid, labels=labels, predictions=predictions)
    write_json(out/"result.json",result)
    note(f"实验 {result['name']}：{result['method']}，切点 {result['cutoff']}，AUC {result['auc']:.6f}，Logloss {result['logloss']:.6f}，耗时 {result['seconds']:.1f} 秒。", "experiment")
    print(json.dumps(result, indent=2), flush=True)


def als_experiment(config):
    from implicit.cpu.als import AlternatingLeastSquares
    out=LAB/"experiments"/config["name"];out.mkdir(parents=True, exist_ok=True)
    (out/"config.json").write_text(json.dumps(config, indent=2))
    start=time.monotonic();cutoff=config.get("cutoff", "2025-01-26")
    path=snapshot(cutoff)
    x=sparse.load_npz(path/"interactions.npz")
    stats=dict(np.load(path/"stats.npz"))
    meta,data=load_cache("data/cache_v1", "train")
    _,valid=temporal_split(data, timestamp(cutoff))
    model=AlternatingLeastSquares(factors=config.get("factors",64), regularization=config.get("regularization",0.1),
                                 alpha=config.get("alpha",20), iterations=config.get("iterations",20),
                                 num_threads=6, random_state=config.get("seed",42))
    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(x, show_progress=False,
                  callback=lambda i,t,l: print(f"ALS iteration {i+1}, {t:.2f}s", flush=True) if (i+1)%5==0 else None)
    items=np.array(data["items"][valid]);users=np.array(data["users"][valid])
    h=np.array(data["history"][valid]);n=len(valid)
    keep=h.ravel()!=0
    q=sparse.coo_matrix((np.ones(keep.sum(),dtype=np.float32),
                        (np.repeat(np.arange(n),20)[keep],h.ravel()[keep])),shape=(n,meta["n_items"])).tocsr()
    q.data=1+np.log(q.data)
    with threadpool_limits(limits=1,user_api="blas"):
        folded=model.recalculate_user(np.arange(n),q)
    trained=model.user_factors[users]
    v=model.item_factors[items]
    raw_known=(trained[:,None,:]*v).sum(2)
    raw_fold=(folded[:,None,:]*v).sum(2)
    pooled=model.item_factors[h].sum(1)/(h!=0).sum(1).clip(1)[:,None]
    raw_pool=(pooled[:,None,:]*v).sum(2)
    known=stats["known_users"][users]
    raw_known[~known]=raw_fold[~known]
    prior=np.log(stats["ctr"]/(1-stats["ctr"]))+stats["ub"][users,None]+stats["ib"][items]
    variants={}
    for feature_name,raw in (("known",raw_known),("fold",raw_fold),("pool",raw_pool),
                             ("blend",0.5*raw_known+0.5*raw_fold)):
        for weight in (1,3,10,30,100):
            variants[f"{feature_name}_w{weight}"]=evaluate(data["labels"][valid],sigmoid(prior+weight*raw))
    best=max(variants,key=lambda k:variants[k]["auc"])
    kind,weight=best.rsplit("_w",1)
    raw={"known":raw_known,"fold":raw_fold,"pool":raw_pool,"blend":0.5*(raw_known+raw_fold)}[kind]
    prediction=sigmoid(prior+float(weight)*raw)
    np.savez(out/"factors.npz",users=model.user_factors,items=model.item_factors)
    np.savez_compressed(out/"raw_scores.npz",known=raw_known,fold=raw_fold,pool=raw_pool,event_indices=valid)
    result={"name":config["name"],"cutoff":cutoff,"method":"implicit_als",**variants[best],
            "best_variant":best,"all_variants":variants,"config":config,"seconds":time.monotonic()-start}
    save_result(out,result,valid,data["labels"][valid],prediction)


def feature_dataset(cutoff="2025-01-26", train_start="2025-01-22", max_events=80000, seed=42, latent_factors=None, use_context=False, sgns_factors=None, sgns_epochs=None, sgns_recipes=None, rolling=False,foldin=None,alignment=None):
    """Features for fitting rows come from an earlier, disjoint snapshot."""
    suffix="" if not latent_factors else "_als"+"_".join(map(str,latent_factors))
    sgns_epochs=sgns_epochs or {str(f):80 if f<=8 else 56 for f in (sgns_factors or [])}
    for key,value in (sgns_recipes or {}).items():
        sgns_epochs[key]=value.get("epochs",sgns_epochs.get(key,56))
    if use_context:suffix+="_context"
    if sgns_factors:
        suffix+="_sgns"+"_".join(map(str,sgns_factors))+"_e56"
        if min(sgns_factors)<=8 and sgns_epochs[str(min(sgns_factors))]==80:suffix+="_lowrank80"
    if sgns_recipes:suffix+="_recipe"+hashlib.sha256(json.dumps(sgns_recipes,sort_keys=True).encode()).hexdigest()[:10]
    if rolling:suffix+="_rolling"
    if foldin:suffix+="_fold"+hashlib.sha256(json.dumps(foldin,sort_keys=True).encode()).hexdigest()[:8]
    if alignment:suffix+='_align'+hashlib.sha256(json.dumps(alignment,sort_keys=True).encode()).hexdigest()[:8]
    folder=LAB/"cache"/f"features_{train_start}_{cutoff}_{max_events}_{seed}{suffix}"
    if sgns_factors and (folder/'metadata.json').exists():
        cached=json.loads((folder/'metadata.json').read_text())
        legacy={str(f):80 if f<=8 and 'lowrank80' in folder.name else 56 for f in sgns_factors}
        if cached.get('sgns_epochs',legacy)!=sgns_epochs:
            # Older human-readable names encode only the common 56/80 cases.
            # An epoch-only experiment must never silently reuse their matrix.
            folder=folder.with_name(folder.name+'_epochs'+hashlib.sha256(json.dumps(sgns_epochs,sort_keys=True).encode()).hexdigest()[:10])
    if (folder/"metadata.json").exists():
        return folder
    folder.mkdir(parents=True,exist_ok=True)
    _,data=load_cache("data/cache_v1","train")
    train_indices=np.flatnonzero((data["timestamps"]>=timestamp(train_start)) &
                                 (data["timestamps"]<timestamp(cutoff)))
    if max_events and len(train_indices)>max_events:
        train_indices=np.sort(np.random.default_rng(seed).choice(train_indices,max_events,replace=False))
    _,valid=temporal_split(data,timestamp(cutoff))
    early=snapshot(train_start);late=snapshot(cutoff)
    print(f"Generate leakage-safe features: {len(train_indices):,} fitting and {len(valid):,} validation events",flush=True)
    from arena_features import complete_features
    config={"latent_factors":latent_factors,"context_features":use_context,"sgns_factors":sgns_factors,
            "sgns_epochs":sgns_epochs,"sgns_recipes":sgns_recipes or {},"foldin":foldin,'alignment':alignment,'train_start':train_start}
    if rolling:
        blocks=[];days=data["timestamps"][train_indices]//86400
        for day in np.unique(days):
            source_day=__import__('datetime').datetime.fromtimestamp(int(day)*86400,__import__('datetime').timezone.utc).strftime("%Y-%m-%d")
            block,cols=complete_features(data,train_indices[days==day],source_day,config)
            if blocks:assert cols==names
            names=cols;blocks.append(block)
        x=np.concatenate(blocks,axis=0)
    else:
        x,names=complete_features(data,train_indices,train_start,config)
    np.save(folder/"train.npy",x);del x
    xv,valid_names=complete_features(data,valid,cutoff,config)
    assert valid_names==names
    np.save(folder/"valid.npy",xv)
    np.savez(folder/"labels.npz",train=data["labels"][train_indices].ravel(),valid=data["labels"][valid].ravel(),
             train_indices=train_indices,validation_indices=valid)
    (folder/"metadata.json").write_text(json.dumps({"feature_names":names,"train_events":len(train_indices),
              "validation_events":len(valid),"train_snapshot":str(early),"validation_snapshot":str(late),
              "latent_factors":latent_factors,"use_context":use_context,"sgns_factors":sgns_factors,"sgns_epochs":sgns_epochs,"sgns_recipes":sgns_recipes,"rolling":rolling,"foldin":foldin,'alignment':alignment},indent=2))
    return folder


def cat_experiment(config):
    from catboost import CatBoostClassifier
    out=LAB/"experiments"/config["name"];out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(config,indent=2))
    start=time.monotonic();cutoff=config.get("cutoff","2025-01-26")
    folder=feature_dataset(cutoff,config.get("train_start","2025-01-22"),config.get("max_events",80000),
                           config.get("data_seed",42),config.get("latent_factors"),config.get("context_features",False),config.get("sgns_factors"),config.get("sgns_epochs"),config.get("sgns_recipes"),config.get("rolling",False),config.get("foldin"),config.get('alignment'))
    info=json.loads((folder/"metadata.json").read_text());labels=np.load(folder/"labels.npz")
    if config.get("sgns_factors"):
        config["sgns_epochs"]=info.get("sgns_epochs",{str(f):80 if f<=8 and "lowrank80" in folder.name else 56 for f in config["sgns_factors"]})
    x=np.load(folder/"train.npy",mmap_mode="r");v=np.load(folder/"valid.npy",mmap_mode="r")
    params=dict(iterations=config.get("iterations",800),depth=config.get("depth",6),learning_rate=config.get("lr",0.06),
                l2_leaf_reg=config.get("l2",10),loss_function="Logloss",eval_metric="AUC",thread_count=6,
                random_seed=config.get("seed",42),early_stopping_rounds=80,allow_writing_files=False,
                border_count=128,bootstrap_type="Bernoulli",subsample=0.8)
    if config.get("fixed_trees",False):
        params.update(early_stopping_rounds=None,use_best_model=False)
    model=CatBoostClassifier(**params)
    model.fit(x,labels["train"],eval_set=(v,labels["valid"]),verbose=100)
    prediction=model.predict_proba(v)[:,1]
    model.save_model(str(out/"model.cbm"))
    importance=sorted(zip(info["feature_names"],model.feature_importances_.tolist()),key=lambda x:-x[1])
    (out/"feature_importance.json").write_text(json.dumps(importance,indent=2))
    result={"name":config["name"],"cutoff":cutoff,"method":"catboost_numeric",**evaluate(labels["valid"],prediction),
            "config":config,"tree_count":model.tree_count_,"seconds":time.monotonic()-start,"feature_dataset":str(folder.relative_to(ROOT)),
            "feature_importance_top10":importance[:10]}
    save_result(out,result,labels["validation_indices"],labels["valid"].reshape(-1,10),prediction.reshape(-1,10))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text())
    if c["method"]=="als":als_experiment(c)
    elif c["method"]=="catboost":cat_experiment(c)
    elif c["method"]=="event":
        from arena_event import event_experiment
        event_experiment(c)
    else:raise ValueError(c["method"])
