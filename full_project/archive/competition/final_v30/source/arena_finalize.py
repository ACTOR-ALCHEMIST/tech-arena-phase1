"""Two-cutoff verification, robust blending, full refit and verified delivery."""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score
from ctr_data import load_cache, temporal_split
from run import timestamp,check_submission
from arena_core import ROOT,LAB,snapshot,features,evaluate,sigmoid
from arena_report import note,render,write_json
from arena_features import complete_features


def explicit_feature_recipe(result):
    config=result.get("feature_config",result.get("config",{}))
    if config.get("sgns_factors") and "sgns_epochs" not in config:
        config["sgns_epochs"]={str(f):80 if f<=8 and "lowrank80" in result.get("feature_dataset","") else 56 for f in config["sgns_factors"]}
    if result.get("method")=="tabular_mlp_mps":result["config"].setdefault("constant_policy","legacy_clip")
    return result


def ensure_secondary(result,audit_cutoff='2025-01-25'):
    result=explicit_feature_recipe(result)
    name=result["name"]+"_locked"+audit_cutoff[-2:]
    folder=LAB/"experiments"/name
    if (folder/"result.json").exists():return json.loads((folder/"result.json").read_text())
    if audit_cutoff=='2025-01-25' and result["name"]=="sgns8_n8_e80_s42" and (LAB/"experiments/sgns8_locked_cut25/result.json").exists():
        return json.loads((LAB/"experiments/sgns8_locked_cut25/result.json").read_text())
    config=dict(result["config"]);config.update(name=name,cutoff=audit_cutoff)
    if "train_start" in config:
        delta=datetime.strptime(result['cutoff'],'%Y-%m-%d')-datetime.strptime(audit_cutoff,'%Y-%m-%d')
        config["train_start"]=(datetime.strptime(config["train_start"],"%Y-%m-%d")-delta).strftime("%Y-%m-%d")
    config["audit_protocol"]="locked hyperparameters, rolling origin, same feature lag"
    note(f"锁定主验证配置，开始 {audit_cutoff} 时间切点复核 {result['name']}。","audit")
    if result["method"]=="catboost_event":
        from arena_event import event_experiment
        base=json.loads((LAB/"experiments"/result["base_experiment"]/"result.json").read_text())
        audited=ensure_secondary(base,audit_cutoff)
        config.update(source_experiment=audited["name"],iterations=result["event_tree_count"],fixed_trees=True,
                      locked_temperature=result["temperature"],locked_mix=result["event_mix"])
        event_experiment(config)
    elif result["method"]=="tabular_mlp_mps":
        from arena_tabular import run
        base=json.loads((LAB/"experiments"/result["source_experiment"]/"result.json").read_text())
        base_audit=LAB/'experiments'/(base['name']+'_locked'+audit_cutoff[-2:])/'result.json'
        if audit_cutoff=='2025-01-24' or not base_audit.exists():
            # A numerical neural head needs the feature matrix, not a fitted
            # intermediate tree. Build the identical frozen feature recipe.
            from arena_models import feature_dataset
            fc=dict(result['feature_config']);fc['cutoff']=audit_cutoff
            delta=datetime.strptime(result['cutoff'],'%Y-%m-%d')-datetime.strptime(audit_cutoff,'%Y-%m-%d')
            fc['train_start']=(datetime.strptime(fc.get('train_start','2025-01-22'),'%Y-%m-%d')-delta).strftime('%Y-%m-%d')
            features_path=feature_dataset(audit_cutoff,fc['train_start'],fc.get('max_events',80000),fc.get('data_seed',42),
                fc.get('latent_factors'),fc.get('context_features',False),fc.get('sgns_factors'),fc.get('sgns_epochs'),
                fc.get('sgns_recipes'),fc.get('rolling',False),fc.get('foldin'),fc.get('alignment'),fc.get('sgns_extra'))
            config['feature_reference']={'name':base['name'],'cutoff':audit_cutoff,'config':fc,
                                         'feature_dataset':str(features_path.relative_to(ROOT))}
            source_name=base['name']
        else:source_name=ensure_secondary(base,audit_cutoff)['name']
        config.update(source_experiment=source_name,epochs=result["best_epoch"],fixed_epochs=True,
                      lr_schedule_epochs=result["config"].get("lr_schedule_epochs",result["config"].get("epochs",30)))
        run(config)
    elif result["method"]=="warm_ctr_mps":
        from arena_warm_ctr import run
        config.update(epochs=result["best_epoch"],fixed_epochs=True)
        run(config)
    elif result["method"]=="catboost_numeric":
        from arena_models import cat_experiment
        config.update(iterations=result["tree_count"],fixed_trees=True)
        cat_experiment(config)
    elif result["method"].startswith("sgns_"):
        from arena_sgns import run
        recipe=next(e for e in result["epochs"] if e["epoch"]==result["best_epoch"])
        config.update(epochs=result["best_epoch"],fixed_epochs=True,locked_variant=recipe["best_variant"],
                      lr_schedule_epochs=result["config"].get("lr_schedule_epochs",result["config"].get("epochs",80)))
        run(config)
    else:raise ValueError("Unsupported audit family")
    return json.loads((folder/"result.json").read_text())


def prediction_record(result):
    values=np.load(LAB/"experiments"/result["name"]/"validation.npz")
    return values["event_indices"],values["labels"],values["predictions"].reshape(-1,10)


def blend(predictions,weights,kind):
    # Dropped components must not change dtype/rounding on single-model recipes.
    pairs=[(w,p) for w,p in zip(weights,predictions) if w>0]
    weights,predictions=zip(*pairs)
    if kind=="probability":return sum(w*p for w,p in zip(weights,predictions))
    logits=[np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7))) for p in predictions]
    return sigmoid(sum(w*z for w,z in zip(weights,logits)))


def select():
    records=[explicit_feature_recipe(json.loads(p.read_text())) for p in (LAB/"experiments").glob("*/result.json")]
    records=[r for r in records if r.get("cutoff")=="2025-01-26"]
    cats=sorted([r for r in records if r["method"].startswith("catboost")],key=lambda r:-r["auc"])
    sg=sorted([r for r in records if r["method"].startswith("sgns_") or r["method"] in ("tabular_mlp_mps","warm_ctr_mps")],key=lambda r:-r["auc"])
    if not cats or not sg:raise ValueError("Need completed tree and GPU representation experiments")
    chosen=[cats[0]]
    for r in cats[1:]:
        if r['method']!=cats[0]['method']:
            chosen.append(r);break
    if len(chosen)==1 and len(cats)>1:chosen.append(cats[1])
    # Keep a small, predeclared pool. Seed averaging can reduce variance without
    # fitting another stacking model to validation labels.
    chosen.extend(sg[:3])
    second=[ensure_secondary(r) for r in chosen]
    arrays=[prediction_record(r) for r in chosen];audits=[prediction_record(r) for r in second]
    for group in (arrays,audits):
        for record in group[1:]:
            np.testing.assert_array_equal(group[0][0],record[0]);np.testing.assert_array_equal(group[0][1],record[1])
    n=len(chosen);weights=[]
    from arena_validation import importance_weights,weighted_auc
    iw_primary,profile_primary=importance_weights("2025-01-26",arrays[0][0])
    iw_secondary,profile_secondary=importance_weights("2025-01-25",audits[0][0])
    for i in range(n):weights.append([float(i==j) for j in range(n)])
    weights += [[1/n]*n]
    from itertools import combinations
    for size in (2,3):
        for subset in combinations(range(n),size):
            weights.append([1/size if j in subset else 0 for j in range(n)])
            if size==2:
                for a in (0.25,0.75):weights.append([a if j==subset[0] else 1-a if j==subset[1] else 0 for j in range(n)])
    neural_ids=[i for i,r in enumerate(chosen) if not r['method'].startswith('catboost')]
    for tree_weight in (0.25,0.5):
        weights.append([tree_weight if j==0 else (1-tree_weight)/len(neural_ids) if j in neural_ids else 0 for j in range(n)])
    scores=[]
    for kind in ("probability","logit"):
        for w in weights:
            primary=blend([a[2] for a in arrays],w,kind)
            secondary=blend([a[2] for a in audits],w,kind)
            a=float(roc_auc_score(arrays[0][1].ravel(),primary.ravel()))
            b=float(roc_auc_score(audits[0][1].ravel(),secondary.ravel()))
            wa=weighted_auc(arrays[0][1],primary,iw_primary);wb=weighted_auc(audits[0][1],secondary,iw_secondary)
            scores.append({"kind":kind,"weights":w,"auc_26":a,"auc_25":b,"weighted_auc_26":wa,
                           "weighted_auc_25":wb,"selection_score":0.5*wa+0.5*wb})
    winner=max(scores,key=lambda r:r["selection_score"])
    profiles={"2025-01-26":profile_primary,"2025-01-25":profile_secondary,
              "criterion":"mean of two cohort-weighted AUCs; unlabeled evaluation covariates only; diagnostic, not official test AUC"}
    (LAB/'selection_cohort_profiles.json').write_text(json.dumps(profiles,indent=2))
    primary=blend([a[2] for a in arrays],winner["weights"],winner["kind"])
    secondary=blend([a[2] for a in audits],winner["weights"],winner["kind"])
    return chosen,second,winner,scores,arrays[0][:2],audits[0][:2],primary,secondary


def frozen_selection(path):
    """Export an already selected recipe without rerunning a wider search."""
    spec=json.loads(Path(path).read_text())
    chosen=[explicit_feature_recipe(json.loads((LAB/'experiments'/name/'result.json').read_text())) for name in spec['sources']]
    second=[ensure_secondary(r) for r in chosen]
    arrays=[prediction_record(r) for r in chosen];audits=[prediction_record(r) for r in second]
    for group in (arrays,audits):
        for record in group[1:]:
            np.testing.assert_array_equal(record[0],group[0][0]);np.testing.assert_array_equal(record[1],group[0][1])
    winner=dict(spec['selected']);weights=winner['weights']
    assert len(weights)==len(chosen) and min(weights)>=0 and abs(sum(weights)-1)<1e-12
    primary=blend([r[2] for r in arrays],weights,winner['kind'])
    secondary=blend([r[2] for r in audits],weights,winner['kind'])
    from arena_validation import importance_weights,weighted_auc
    for suffix,cutoff,records,pred in [('26','2025-01-26',arrays,primary),('25','2025-01-25',audits,secondary)]:
        winner['auc_'+suffix]=evaluate(records[0][1],pred)['auc']
        iw,_=importance_weights(cutoff,records[0][0])
        winner['weighted_auc_'+suffix]=weighted_auc(records[0][1],pred,iw)
    winner['selection_score']=(winner['weighted_auc_26']+winner['weighted_auc_25'])/2
    return chosen,second,winner,spec.get('all_recipes',[winner]),arrays[0][:2],audits[0][:2],primary,secondary


def write_feature_matrix(data,indices,cutoff,config,target,rolling=False):
    if target.exists():return np.load(target,mmap_mode="r")
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_name(target.name+".partial")
    matrix=None
    for offset in range(0,len(indices),15000):
        chosen=indices[offset:offset+15000]
        if rolling:
            parts=[];days=data["timestamps"][chosen]//86400
            for day in np.unique(days):
                source_day=datetime.fromtimestamp(int(day)*86400,ZoneInfo("UTC")).strftime("%Y-%m-%d")
                block,cols=complete_features(data,chosen[days==day],source_day,config)
                if parts:assert cols==names
                names=cols;parts.append(block)
            x=np.concatenate(parts,axis=0)
        else:x,names=complete_features(data,chosen,cutoff,config)
        if matrix is None:
            matrix=np.lib.format.open_memmap(partial,mode="w+",dtype=np.float32,shape=(len(indices)*10,len(names)))
            target.with_suffix(".columns.json").write_text(json.dumps(names))
        matrix[offset*10:(offset+len(chosen))*10]=x
        matrix.flush()
        print(f"final feature matrix {offset+len(chosen):,}/{len(indices):,} events",flush=True)
    del matrix
    os.replace(partial,target)
    return np.load(target,mmap_mode="r")


def feature_cache(config,phase):
    keys=("latent_factors","context_features","sgns_factors","sgns_epochs","sgns_recipes","foldin",'alignment')
    spec={key:config.get(key) for key in keys}
    if config.get('sgns_extra'):spec['sgns_extra']=config['sgns_extra']
    if phase=="train":spec.update(train_start=config.get("train_start","2025-01-22"),rolling=config.get("rolling",False))
    if config.get('alignment'):spec['alignment_anchor']=config.get('train_start','2025-01-22')
    meta,_=load_cache("data/cache_v1","train");spec["cache_id"]=meta["cache_id"]
    if phase=='evaluation' and os.environ.get('ARENA_EVALUATION_CACHE_TAG'):
        spec['replay_tag']=os.environ['ARENA_EVALUATION_CACHE_TAG']
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
    return LAB/"cache"/f"final_{phase}_{digest}.npy"


def refit_cat(result,folder):
    from catboost import CatBoostClassifier
    folder.mkdir(parents=True,exist_ok=True);config=dict(result["config"])
    if (folder/"model.cbm").exists() and (folder/"recipe.json").exists():return
    meta,data=load_cache("data/cache_v1","train")
    start=config.get("train_start","2025-01-22")
    indices=np.flatnonzero(data["timestamps"]>=timestamp(start))
    target=feature_cache(config,"train")
    x=write_feature_matrix(data,indices,start,config,target,rolling=config.get("rolling",False))
    source_info=json.loads((ROOT/result["feature_dataset"]/"metadata.json").read_text())
    columns=json.loads(target.with_suffix(".columns.json").read_text())
    assert columns==source_info["feature_names"],"Feature schema changed since validation"
    model=CatBoostClassifier(iterations=result["tree_count"],depth=config.get("depth",6),
        learning_rate=config.get("lr",0.06),l2_leaf_reg=config.get("l2",10),loss_function="Logloss",
        thread_count=6,random_seed=config.get("seed",42),border_count=128,bootstrap_type="Bernoulli",
        subsample=0.8,allow_writing_files=False)
    model.fit(x,data["labels"][indices].ravel(),verbose=100)
    model.save_model(str(folder/"model.cbm"))
    recipe={"type":"catboost","source_experiment":result["name"],"config":config,"tree_count":result["tree_count"],
            "training_events":len(indices),"representation_cutoff":"2025-01-27","feature_names":columns,"cache_id":meta["cache_id"]}
    if result["method"]=="catboost_event":
        from arena_event import aggregate
        gate=CatBoostClassifier(iterations=result["event_tree_count"],depth=5,learning_rate=0.04,l2_leaf_reg=20,
                               loss_function="Logloss",thread_count=6,random_seed=42,allow_writing_files=False)
        gate.fit(aggregate(x),data["labels"][indices].sum(1),verbose=100)
        gate.save_model(str(folder/"event_gate.cbm"))
        recipe.update(event_temperature=result["temperature"],event_mix=result["event_mix"])
    (folder/"recipe.json").write_text(json.dumps(recipe,indent=2))


def refit_sgns(result,folder):
    from arena_sgns import run
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"training_result.json").exists() and (folder/"recipe.json").exists():return
    config=dict(result["config"])
    config.update(name="full_"+result["name"],cutoff="2025-01-27",epochs=result["best_epoch"],evaluate=False,
                  output_dir=str(folder),lr_schedule_epochs=result["config"].get("lr_schedule_epochs",result["config"].get("epochs",80)))
    run(config)
    recipe=next(e for e in result["epochs"] if e["epoch"]==result["best_epoch"])
    (folder/"recipe.json").write_text(json.dumps({"type":"sgns","source_experiment":result["name"],
          "config":config,"variant":recipe["best_variant"],"representation_cutoff":"2025-01-27"},indent=2))


def refit_tabular(result,folder):
    import copy
    import torch
    from torch.nn import functional as F
    from arena_tabular import build_model,normalizer
    folder.mkdir(parents=True,exist_ok=True)
    config=result["config"];feature_config=result["feature_config"]
    _,data=load_cache("data/cache_v1","train")
    start=feature_config.get("train_start","2025-01-22");indices=np.flatnonzero(data["timestamps"]>=timestamp(start))
    target=feature_cache(feature_config,"train")
    x=write_feature_matrix(data,indices,start,feature_config,target,rolling=feature_config.get("rolling",False))
    columns=json.loads(target.with_suffix(".columns.json").read_text())
    source_columns=json.loads((ROOT/result["feature_dataset"]/"metadata.json").read_text())["feature_names"]
    assert columns==source_columns
    y=data["labels"][indices];mean,scale=normalizer(x,config.get("constant_policy","legacy_clip"))
    torch.manual_seed(config.get("seed",42));torch.set_num_threads(2);device=torch.device("mps")
    loss_name=config.get("loss","event_ce");model=build_model(x.shape[1],mean,scale,config).to(device)
    ema=copy.deepcopy(model).eval() if config.get("ema") else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get("lr",0.001),weight_decay=config.get("weight_decay",0.01))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.get("lr_schedule_epochs",config.get("epochs",30)),eta_min=0.0001)
    rng=np.random.default_rng(config.get("seed",42));records=[]
    for epoch in range(1,result["best_epoch"]+1):
        model.train();order=rng.permutation(len(indices));total=0
        batch_events=config.get('batch_events',1024)
        for offset in range(0,len(order),batch_events):
            chosen=order[offset:offset+batch_events];rows=(chosen[:,None]*10+np.arange(10)).ravel()
            bx=torch.tensor(np.array(x[rows]).reshape(-1,10,x.shape[1]),device=device)
            by=torch.tensor(y[chosen],dtype=torch.float32,device=device);z=model(bx)
            loss=F.cross_entropy(z,torch.where(by.sum(1)>0,by.argmax(1),10)) if loss_name=="event_ce" else F.binary_cross_entropy_with_logits(z,by)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=loss.item()*len(chosen)
            if ema is not None:
                with torch.no_grad():
                    for target,parameter in zip(ema.parameters(),model.parameters()):target.mul_(config["ema"]).add_(parameter,alpha=1-config["ema"])
        scheduler.step();records.append({"epoch":epoch,"loss":total/len(indices)});print("final neural",records[-1],flush=True)
    torch.save({"state_dict":(ema if ema is not None else model).state_dict(),"width":x.shape[1],"loss":loss_name,"config":config},folder/"model.pt")
    (folder/"training_result.json").write_text(json.dumps(records,indent=2))
    (folder/"recipe.json").write_text(json.dumps({"type":"tabular","config":feature_config,"loss":loss_name,
             "width":x.shape[1],"dropout":config.get("dropout",0.1),"architecture":config.get('architecture','point'),
             "training_events":len(indices),"parameters":sum(p.numel() for p in model.parameters()),
             "feature_names":columns,"source_experiment":result["name"]},indent=2))


def refit_warm(result,folder):
    from arena_warm_ctr import run
    config=dict(result["config"]);config.update(output_dir=str(folder),cutoff="2025-01-27",epochs=result["best_epoch"],evaluate=False)
    run(config)
    (folder/"recipe.json").write_text(json.dumps({"type":"warm","config":config,"source_experiment":result["name"]},indent=2))


def infer_model(folder,data):
    recipe=json.loads((folder/"recipe.json").read_text());indices=np.arange(len(data["users"]))
    if recipe["type"] in ("catboost","tabular"):
        from catboost import CatBoostClassifier
        target=feature_cache(recipe["config"],"evaluation")
        x=write_feature_matrix(data,indices,"2025-01-27",recipe["config"],target)
        names=json.loads(target.with_suffix(".columns.json").read_text())
        assert names==recipe["feature_names"]
        if recipe["type"]=="tabular":
            import torch
            from arena_tabular import build_model,predict
            saved=torch.load(folder/"model.pt",map_location="cpu",weights_only=True);state=saved["state_dict"]
            model_config=saved.get('config',dict(loss=saved['loss'],dropout=recipe['dropout'],architecture=recipe.get('architecture','point')))
            model=build_model(saved["width"],state["mean"].numpy(),state["scale"].numpy(),model_config)
            model.load_state_dict(state);device=torch.device("mps" if torch.backends.mps.is_available() else "cpu");model.to(device)
            return predict(model,x,device)
        assert names==recipe["feature_names"]
        model=CatBoostClassifier();model.load_model(str(folder/"model.cbm"))
        p=model.predict_proba(x,thread_count=6)[:,1].reshape(-1,10)
        if "event_temperature" in recipe:
            from arena_event import aggregate,event_probability
            gate=CatBoostClassifier();gate.load_model(str(folder/"event_gate.cbm"))
            p=event_probability(p,gate.predict_proba(aggregate(x))[:,1],recipe["event_temperature"],recipe["event_mix"])
        return p
    if recipe["type"]=="warm":
        import torch
        from arena_warm_ctr import WarmCTR
        from run import predict_arrays
        saved=torch.load(folder/"model.pt",map_location="cpu",weights_only=True);s=saved["state_dict"]
        pretrained={"users":s["u0"].numpy(),"items":s["v0"].numpy()}
        priors=(float(torch.sigmoid(s["global_bias"])),s["ub0"].numpy().ravel()/0.3,s["ib0"].numpy().ravel()/0.3,s["known_users"].numpy(),s["known_items"].numpy())
        c=recipe["config"];model=WarmCTR(pretrained,priors,c.get("anchor",0.1),c.get("user_dropout",0.1),c.get("fine_tune",True));model.load_state_dict(s)
        device=torch.device("mps" if torch.backends.mps.is_available() else "cpu");model.to(device)
        return predict_arrays(model,data,indices,1024,device)
    factors=np.load(folder/"factors.npz");u=factors["users"][data["users"]];v=factors["items"][data["items"]]
    raw=np.einsum("bik,bk->bi",v,u)
    kind,weight=recipe["variant"].rsplit("_prior",1)
    if kind in ("usernorm","cosine"):raw/=np.linalg.norm(u,axis=1).clip(1e-6)[:,None]
    if kind=="cosine":raw/=np.linalg.norm(v,axis=2).clip(1e-6)
    stats=np.load(snapshot("2025-01-27")/"stats.npz")
    prior=np.log(stats["ctr"]/(1-stats["ctr"]))+stats["ub"][data["users"],None]+stats["ib"][data["items"]]
    return sigmoid(-3+0.5*(raw/max(float(raw.std()),1e-6)+float(weight)*prior))


def export_csv(path,data,prediction):
    import csv
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f);writer.writerow(["rowid","predicted_probability"])
        writer.writerows((rid.decode(),f"{float(p):.10g}") for rid,p in zip(data["rowids"].ravel(),prediction.ravel()))
    check_submission(path,"evaluation.zip")


def main():
    p=argparse.ArgumentParser();p.add_argument("--predict-only",action="store_true")
    p.add_argument('--final-dir',default='competition/final')
    p.add_argument('--selection-file');p.add_argument('--reuse-model-dir')
    a=p.parse_args();final=(ROOT/a.final_dir).resolve();final.mkdir(parents=True,exist_ok=True)
    if a.predict_only:
        manifest=json.loads((final/"manifest.json").read_text());version=ROOT/manifest["model_dir"]
        if (version/'representations').exists():os.environ['ARENA_SNAPSHOT_ROOT']=str(version/'representations')
        _,data=load_cache("data/cache_v1","evaluation")
        predictions=[infer_model(version/"models"/str(i),data) for i in range(len(manifest["sources"]))]
        export_csv(version/"reproduced_submission.csv",data,blend(predictions,manifest["weights"],manifest["blend_kind"]))
        return
    chosen,second,winner,scores,primary_data,secondary_data,primary_pred,secondary_pred=frozen_selection(a.selection_file) if a.selection_file else select()
    if final==LAB/'final' and (final/'manifest.json').exists():
        incumbent=json.loads((final/'manifest.json').read_text())
        if any(winner['auc_'+day]<incumbent['validation_'+day]['auc'] for day in ('26','25')) or winner['selection_score']<incumbent['selection_score']:
            raise ValueError('Candidate would regress the frozen incumbent; export to a separate --final-dir for review')
    all_sources=[r["name"] for r in chosen]
    active=[i for i,w in enumerate(winner["weights"]) if w>0]
    chosen=[chosen[i] for i in active];second=[second[i] for i in active]
    winner=dict(winner);winner["weights"]=[winner["weights"][i] for i in active]
    version=final/"versions"/datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    version.mkdir(parents=True)
    (version/"selection.json").write_text(json.dumps({"selected":winner,"all_recipes":scores,
              "sources":[r["name"] for r in chosen],"all_candidate_sources":all_sources,"secondary":[r["name"] for r in second]},indent=2))
    np.savez_compressed(version/"validation_26.npz",event_indices=primary_data[0],labels=primary_data[1],predictions=primary_pred)
    np.savez_compressed(version/"validation_25.npz",event_indices=secondary_data[0],labels=secondary_data[1],predictions=secondary_pred)
    third=[ensure_secondary(result,'2025-01-24') for result in chosen]
    third_arrays=[prediction_record(result) for result in third]
    for array in third_arrays[1:]:
        np.testing.assert_array_equal(array[0],third_arrays[0][0]);np.testing.assert_array_equal(array[1],third_arrays[0][1])
    third_prediction=blend([array[2] for array in third_arrays],winner['weights'],winner['kind'])
    np.savez_compressed(version/'validation_24.npz',event_indices=third_arrays[0][0],labels=third_arrays[0][1],predictions=third_prediction)
    note(f"未参与参数和权重选择的 Jan24 回溯复核 AUC {evaluate(third_arrays[0][1],third_prediction)['auc']:.6f}；冻结配方继续全量重训。",'audit')
    note(f"双切点选择配方：{winner}，来源 {[r['name'] for r in chosen]}；开始全量重训。","finalize")
    reusable={}
    if a.reuse_model_dir:
        for folder in (ROOT/a.reuse_model_dir/'models').iterdir():
            if (folder/'recipe.json').exists():
                reusable[json.loads((folder/'recipe.json').read_text())['source_experiment']]=folder
    for i,result in enumerate(chosen):
        if result['name'] in reusable:
            shutil.copytree(reusable[result['name']],version/'models'/str(i));continue
        if result["method"].startswith("catboost"):refit_cat(result,version/"models"/str(i))
        elif result["method"]=="tabular_mlp_mps":refit_tabular(result,version/"models"/str(i))
        elif result["method"]=="warm_ctr_mps":refit_warm(result,version/"models"/str(i))
        else:refit_sgns(result,version/"models"/str(i))
    # Keep the learned representations with the winning heads, so they cannot
    # accidentally disappear when intermediate experiment caches are cleaned.
    representations=version/'representations';representations.mkdir()
    dates={'2025-01-27'}
    for result in chosen:
        fc=result.get('feature_config',result['config'])
        if fc.get('alignment'):dates.add(fc['train_start'])
    for day in dates:
        shutil.copytree(snapshot(day),representations/('snapshot_'+day),ignore=shutil.ignore_patterns('*.lock','*.tmp'))
    os.environ['ARENA_SNAPSHOT_ROOT']=str(representations)
    _,data=load_cache("data/cache_v1","evaluation")
    predictions=[infer_model(version/"models"/str(i),data) for i in range(len(chosen))]
    prediction=blend(predictions,winner["weights"],winner["kind"])
    export_csv(version/"submission.csv",data,prediction)
    shutil.copy2(version/"submission.csv",final/"submission.csv.partial")
    os.replace(final/"submission.csv.partial",final/"submission.csv")
    manifest={"submission":str((final/'submission.csv').relative_to(ROOT)),"model_dir":str(version.relative_to(ROOT)),
              "sources":[r["name"] for r in chosen],"weights":winner["weights"],"blend_kind":winner["kind"],
              "validation_26":evaluate(primary_data[1],primary_pred),"validation_25":evaluate(secondary_data[1],secondary_pred),
              "validation_24":evaluate(third_arrays[0][1],third_prediction),'audit_24_sources':[r['name'] for r in third],
              'validation_protocol':'Jan26 used for hyperparameters; Jan26+Jan25 cohort-weighted AUC selects coarse blend; Jan24 recipe audit is not used for selection',
              "rows":int(prediction.size),"sha256":hashlib.sha256((final/"submission.csv").read_bytes()).hexdigest(),
              "csv_checks_passed":True,"official_evaluation_labels_available":False,
              "selection_score":winner["selection_score"],"cache_dependencies":"data/cache_v1 holds ID mappings and cached event arrays; learned inference representations are bundled in the version directory; competition/cache holds regenerable feature matrices",
              "representations":str(representations.relative_to(ROOT)),
              "reproduce":f"python3 arena_finalize.py --final-dir {final.relative_to(ROOT)} --predict-only"}
    write_json(final/"manifest.json",manifest)
    sources=version/"source";sources.mkdir()
    for name in ROOT.glob("*.py"):shutil.copy2(name,sources/name.name)
    shutil.copy2(ROOT/"overnight.go",sources/"overnight.go")
    shutil.copy2(ROOT/'overnight_test.go',sources/'overnight_test.go')
    for name in ('INSTRUCTIONS.md','requirements.txt','requirements-arena.txt'):
        shutil.copy2(ROOT/name,sources/name)
    from importlib.metadata import version as package_version
    environment={'python':__import__('sys').version,'platform':__import__('platform').platform(),
                 'packages':{name:package_version(name) for name in ('numpy','scipy','scikit-learn','torch','numba','catboost','implicit')},
                 'go':__import__('subprocess').check_output([str(ROOT/'.tools/go/bin/go'),'version'],text=True).strip()}
    import torch,psutil
    environment.update(mps_available=torch.backends.mps.is_available(),cpu_cores=os.cpu_count(),ram_bytes=psutil.virtual_memory().total)
    (version/'environment.json').write_text(json.dumps(environment,indent=2))
    if (LAB/'selection_cohort_profiles.json').exists():shutil.copy2(LAB/'selection_cohort_profiles.json',version/'cohort_profiles.json')
    note(f"最终 CSV 导出并检查通过，{prediction.size:,} 行；主验证 AUC {manifest['validation_26']['auc']:.6f}，第二切点 {manifest['validation_25']['auc']:.6f}。","delivery")
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":main()
