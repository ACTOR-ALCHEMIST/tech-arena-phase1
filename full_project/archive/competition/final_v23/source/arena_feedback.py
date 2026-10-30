"""Recover distinct observed clicks and context statistics without label leakage."""
import json
import os
import numpy as np
from ctr_data import load_cache
from run import timestamp
from arena_core import snapshot


def feedback(cutoff):
    folder=snapshot(cutoff);target=folder/"distinct_clicks.npz"
    if target.exists():return dict(np.load(target))
    meta,data=load_cache("data/cache_v1","train")
    fit=np.flatnonzero(data["timestamps"]<timestamp(cutoff))
    h=np.array(data["history"][fit]);valid=h.ravel()!=0
    u=np.repeat(data["users"][fit],20)[valid].astype(np.int64)
    item=h.ravel()[valid].astype(np.int64)
    minutes=(data["timestamps"][fit,None]//60-np.rint(data["age_hours"][fit]*60).astype(np.int64)).ravel()[valid]
    clicked_r,clicked_c=np.nonzero(data["labels"][fit])
    u=np.concatenate([u,data["users"][fit[clicked_r]]])
    item=np.concatenate([item,data["items"][fit[clicked_r],clicked_c]])
    minutes=np.concatenate([minutes,data["timestamps"][fit[clicked_r]]//60])
    base=int(minutes.min());stride=int(minutes.max()-base+1)
    keys=np.unique((u*meta["n_items"]+item)*stride+(minutes-base))
    pair=keys//stride
    result={"users":(pair//meta["n_items"]).astype(np.int32),
            "items":(pair%meta["n_items"]).astype(np.int32),
            "minutes":(keys%stride+base).astype(np.int32)}
    assert int(result["minutes"].max())*60<timestamp(cutoff)
    temp=target.with_name(target.name+f".{os.getpid()}.tmp")
    with temp.open("wb") as handle:np.savez_compressed(handle,**result)
    os.replace(temp,target)
    (folder/"feedback_metadata.json").write_text(json.dumps({"distinct_observed_clicks":len(keys),
        "deduplication":"user_id,item_id,click_timestamp_minute","source":"only fitting events and their prior histories"},indent=2))
    print(f"Recovered {len(keys):,} distinct past clicks before {cutoff}",flush=True)
    return result


def context_stats(cutoff):
    folder=snapshot(cutoff);target=folder/"context_stats.npz"
    if target.exists():return dict(np.load(target))
    meta,data=load_cache("data/cache_v1","train");obs=feedback(cutoff)
    u,i,minutes=obs["users"],obs["items"],obs["minutes"]
    bucket=(minutes%1440)//360
    # Unix day zero is Thursday. UTC weekend is weekday >= 5.
    day=(((minutes//1440+3)%7)>=5).astype(np.int32)
    result={}
    for name,ids,size in (("user",u,meta["n_users"]),("item",i,meta["n_items"])):
        total=np.bincount(ids,minlength=size)
        result[name+"_click_support"]=total.astype(np.float32)
        for label,cond,k in (("bucket",bucket,4),("weekend",day,2)):
            cnt=np.bincount(ids.astype(np.int64)*k+cond,minlength=size*k).reshape(size,k)
            prior=np.bincount(cond,minlength=k)/len(cond)
            result[name+"_"+label+"_affinity"]=np.log(((cnt+5*prior)/(total[:,None]+5))/prior).astype(np.float32)
    fit=np.flatnonzero(data["timestamps"]<timestamp(cutoff));labels=data["labels"][fit]
    bucket=((data["timestamps"][fit]%86400)//21600).astype(np.int32)
    ids=np.array(data["items"][fit]);users=np.array(data["users"][fit])
    for name,item_ids,size,y,exposure in (("user",users,meta["n_users"],labels.sum(1),10),
                                      ("item",ids.ravel(),meta["n_items"],labels.ravel(),1)):
        cond=bucket if name=="user" else np.repeat(bucket,10)
        keys=item_ids.astype(np.int64)*4+cond
        impr=np.bincount(keys,minlength=size*4).reshape(size,4)*exposure
        clicks=np.bincount(keys,weights=y,minlength=size*4).reshape(size,4)
        ctr=(clicks.sum(1)+50*float(labels.mean()))/(impr.sum(1)+50)
        rate=(clicks+30*ctr[:,None])/(impr+30)
        result[name+"_bucket_ctr_offset"]=(np.log(rate/(1-rate))-np.log(ctr/(1-ctr))[:,None]).astype(np.float32)
    temp=target.with_name(target.name+f".{os.getpid()}.tmp")
    with temp.open("wb") as handle:np.savez(handle,**result)
    os.replace(temp,target)
    return result


def context_features(data,indices,cutoff):
    stats=context_stats(cutoff)
    items=np.array(data["items"][indices]);users=np.array(data["users"][indices])
    ts=np.array(data["timestamps"][indices]);bucket=((ts%86400)//21600).astype(np.int32)
    weekend=np.array(data["context"][indices,2]).astype(np.int32)
    arrays=[];names=[]
    for name,ids in (("user",users),("item",items)):
        for label,cond in (("bucket",bucket),("weekend",weekend)):
            values=stats[name+"_"+label+"_affinity"][ids,cond if name=="user" else cond[:,None]]
            if name=="user":values=np.broadcast_to(values[:,None],items.shape)
            arrays.append(values);names.append(name+"_"+label+"_affinity")
        values=stats[name+"_bucket_ctr_offset"][ids,bucket if name=="user" else bucket[:,None]]
        if name=="user":values=np.broadcast_to(values[:,None],items.shape)
        arrays.append(values);names.append(name+"_bucket_ctr_offset")
    h=np.array(data["history"][indices]);age=np.array(data["age_hours"][indices])
    hminutes=ts[:,None]//60-np.rint(age*60).astype(np.int64)
    same=(hminutes%1440)//360==bucket[:,None]
    valid=h!=0
    arrays.append((same&valid).sum(1)[:,None].repeat(10,axis=1)/20);names.append("history_same_bucket_fraction")
    match=(items[:,:,None]==h[:,None,:])&valid[:,None,:]&same[:,None,:]
    arrays.append(match.sum(2));names.append("repeat_same_bucket_count")
    arrays.append(np.log1p(ts%86400/3600)[:,None].repeat(10,axis=1));names.append("hour_log")
    return np.stack(arrays,axis=2).reshape(-1,len(names)).astype(np.float32),names
