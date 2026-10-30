"""GPU auxiliary implicit-feedback MF with explicit sampled-negative objective."""
import argparse
import json
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import sparse
from ctr_data import load_cache, temporal_split
from run import timestamp
from arena_core import snapshot, evaluate, sigmoid
from arena_report import LAB, note,write_json


def run(config):
    torch.manual_seed(config.get("seed",42));torch.set_num_threads(2)
    device=torch.device(config.get("device","mps"))
    if device.type=="mps" and not torch.backends.mps.is_available():raise RuntimeError("MPS unavailable")
    start=time.monotonic();cutoff=config.get("cutoff","2025-01-26")
    out=__import__('pathlib').Path(config["output_dir"]) if config.get("output_dir") else LAB/"experiments"/config["name"]
    out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(config,indent=2))
    path=snapshot(cutoff);x=sparse.load_npz(path/"interactions.npz");stats=dict(np.load(path/"stats.npz"))
    user_ids,item_ids=x.nonzero();n=len(user_ids);dim=config.get("factors",16)
    if config.get("feedback_mode")=="distinct_clicks":
        from arena_feedback import feedback
        observed=feedback(cutoff);user_ids=observed["users"];item_ids=observed["items"];n=len(user_ids)
    u=nn.Embedding(x.shape[0],dim,padding_idx=0,device=device)
    v=nn.Embedding(x.shape[1],dim,padding_idx=0,device=device)
    bias=nn.Embedding(x.shape[1],1,padding_idx=0,device=device)
    nn.init.normal_(u.weight,std=0.1);nn.init.normal_(v.weight,std=0.1);nn.init.zeros_(bias.weight)
    with torch.no_grad():u.weight[0].zero_();v.weight[0].zero_()
    optimizer=torch.optim.AdamW(list(u.parameters())+list(v.parameters())+list(bias.parameters()),
                                lr=config.get("lr",0.02),weight_decay=config.get("weight_decay",0.01))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.get("lr_schedule_epochs",config.get("epochs",20)),eta_min=0.001) if config.get("lr_decay",False) else None
    ids_u=torch.tensor(user_ids,dtype=torch.long,device=device)
    ids_i=torch.tensor(item_ids,dtype=torch.long,device=device)
    do_evaluate=config.get("evaluate",True)
    if do_evaluate:
        meta,data=load_cache("data/cache_v1","train");_,valid=temporal_split(data,timestamp(cutoff))
        vi=np.array(data["items"][valid]);vu=np.array(data["users"][valid]);labels=data["labels"][valid]
        prior=np.log(stats["ctr"]/(1-stats["ctr"]))+stats["ub"][vu,None]+stats["ib"][vi]
    negative=config.get("negatives",8);batch_size=config.get("batch_size",8192)
    best=-1;records=[];best_prediction=None;stale=0
    for epoch in range(1,config.get("epochs",20)+1):
        t=time.monotonic();order=torch.randperm(n,device=device);losses=[]
        for offset in range(0,n,batch_size):
            chosen=order[offset:offset+batch_size]
            uid=ids_u[chosen];iid=ids_i[chosen]
            nid=torch.randint(1,x.shape[1],(len(chosen),negative),device=device)
            uv=u(uid);pv=v(iid);nv=v(nid)
            pos=(uv*pv).sum(1)+bias(iid).squeeze(1)-math.log(negative)
            neg=(uv[:,None,:]*nv).sum(2)+bias(nid).squeeze(2)-math.log(negative)
            loss=(F.softplus(-pos).mean()+negative*F.softplus(neg).mean())/(negative+1)
            loss=loss+config.get("regularization",0.001)*(uv.square().mean()+pv.square().mean()+nv.square().mean())
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(loss.item())
        if device.type=="mps":torch.mps.synchronize()
        if scheduler is not None:scheduler.step()
        uf=u.weight.detach().cpu().numpy();vf=v.weight.detach().cpu().numpy();bf=bias.weight.detach().cpu().numpy().ravel()
        if not do_evaluate or (epoch%config.get("eval_every",1)!=0 and epoch!=config.get("epochs",20)):
            row={"epoch":epoch,"loss":float(np.mean(losses)),"seconds":time.monotonic()-t}
            if not do_evaluate:records.append(row)
            if epoch%10==0 or epoch==config.get("epochs",20):
                print(json.dumps(row),flush=True)
                if not do_evaluate:
                    np.savez(out/"factors.npz",users=uf,items=vf,bias=bf)
                    (out/"training.json").write_text(json.dumps(records,indent=2))
            continue
        raw=np.einsum("bik,bk->bi",vf[vi],uf[vu])
        un=np.linalg.norm(uf[vu],axis=1).clip(1e-6);vn=np.linalg.norm(vf[vi],axis=2).clip(1e-6)
        scores={"dot":raw,"usernorm":raw/un[:,None],"cosine":raw/(un[:,None]*vn)}
        variants={};predictions={}
        for name,score in scores.items():
            scale=max(float(score.std()),1e-6)
            for weight in (0,0.2,0.5):
                key=f"{name}_prior{weight}"
                if config.get("locked_variant") and key!=config["locked_variant"]:continue
                p=sigmoid(-3+0.5*(score/scale+weight*prior))
                predictions[key]=p;variants[key]=evaluate(labels,p)
        winner=config.get("locked_variant") or max(variants,key=lambda key:variants[key]["auc"])
        row={"epoch":epoch,"loss":float(np.mean(losses)),"best_variant":winner,"auc":variants[winner]["auc"],
             "seconds":time.monotonic()-t}
        records.append(row);print(json.dumps(row),flush=True)
        should_save=(epoch==config.get("epochs",20)) if config.get("fixed_epochs",False) else row["auc"]>best
        if should_save:
            best=row["auc"];stale=0;best_prediction=predictions[winner]
            np.savez(out/"factors.npz",users=uf,items=vf,bias=bf)
            (out/"best_variants.json").write_text(json.dumps(variants,indent=2))
            raw_score=scores[winner.rsplit("_prior",1)[0]]
            (out/"prediction_recipe.json").write_text(json.dumps({"variant":winner,"score_std":float(raw_score.std()),"epoch":epoch},indent=2))
        else:stale+=1
        (out/"training.json").write_text(json.dumps(records,indent=2))
        if not config.get("fixed_epochs",False) and stale>=config.get("patience",8):break
    if not do_evaluate:
        (out/"training_result.json").write_text(json.dumps({"config":config,"seconds":time.monotonic()-start,"epochs":records},indent=2))
        return
    np.savez_compressed(out/"validation.npz",event_indices=valid,labels=labels,predictions=best_prediction)
    result={"name":config["name"],"cutoff":cutoff,"method":"sgns_"+device.type,**evaluate(labels,best_prediction),
            "config":config,"best_epoch":config["epochs"] if config.get("fixed_epochs",False) else max(records,key=lambda r:r["auc"])["epoch"],"epochs":records,"seconds":time.monotonic()-start}
    write_json(out/"result.json",result)
    note(f"GPU 辅助反馈实验 {config['name']}：设备 {device}，AUC {result['auc']:.6f}，最佳 epoch {result['best_epoch']}，耗时 {result['seconds']:.1f} 秒。此负采样仅用于辅助表示，不冒充真实曝光标签。","experiment")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args();run(json.loads(__import__('pathlib').Path(a.config).read_text()))
