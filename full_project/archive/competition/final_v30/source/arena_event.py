"""Factor event engagement from within-event choice; respect the no-click option."""
import json
import shutil
import time
import numpy as np
from arena_core import LAB,ROOT,evaluate
from arena_report import note,write_json
from catboost import CatBoostClassifier


def aggregate(x):
    grouped=np.asarray(x).reshape(-1,10,x.shape[1])
    return np.concatenate([grouped.mean(1),grouped.max(1),grouped.std(1)],axis=1)


def event_probability(point,gate,temperature,mix):
    z=np.log(np.clip(point,1e-7,1-1e-7)/(1-np.clip(point,1e-7,1-1e-7)))/temperature
    z-=z.max(1,keepdims=True);conditional=np.exp(z);conditional/=conditional.sum(1,keepdims=True)
    return (1-mix)*point+mix*gate[:,None]*conditional


def event_experiment(config):
    start=time.monotonic();source=LAB/"experiments"/config["source_experiment"]
    base=json.loads((source/"result.json").read_text())
    if base["config"].get("sgns_factors") and "sgns_epochs" not in base["config"]:
        base["config"]["sgns_epochs"]={str(f):80 if f<=8 and "lowrank80" in base["feature_dataset"] else 56 for f in base["config"]["sgns_factors"]}
    folder=ROOT/base["feature_dataset"];labels=np.load(folder/"labels.npz")
    x=np.load(folder/"train.npy",mmap_mode="r");v=np.load(folder/"valid.npy",mmap_mode="r")
    event_y=labels["train"].reshape(-1,10).sum(1);event_v=labels["valid"].reshape(-1,10).sum(1)
    gate=CatBoostClassifier(iterations=config.get("iterations",1000),depth=5,learning_rate=0.04,
            l2_leaf_reg=20,loss_function="Logloss",eval_metric="Logloss",thread_count=6,random_seed=42,
            allow_writing_files=False,early_stopping_rounds=None if config.get("fixed_trees") else 100,
            use_best_model=not config.get("fixed_trees",False))
    gate.fit(aggregate(x),event_y,eval_set=(aggregate(v),event_v),verbose=100)
    gate_p=gate.predict_proba(aggregate(v))[:,1]
    original=np.load(source/"validation.npz")["predictions"].reshape(-1,10)
    variants={}
    for temperature in ([config["locked_temperature"]] if "locked_temperature" in config else [0.5,0.75,1,1.5,2,3]):
        for mix in ([config["locked_mix"]] if "locked_mix" in config else [0.25,0.5,0.75,1]):
            p=event_probability(original,gate_p,temperature,mix)
            variants[(temperature,mix)]=evaluate(labels["valid"],p)
    temperature,mix=max(variants,key=lambda key:variants[key]["auc"])
    prediction=event_probability(original,gate_p,temperature,mix)
    out=LAB/"experiments"/config["name"];out.mkdir(parents=True,exist_ok=True)
    gate.save_model(str(out/"event_gate.cbm"));shutil.copy2(source/"model.cbm",out/"model.cbm")
    np.savez_compressed(out/"validation.npz",event_indices=labels["validation_indices"],
                       labels=labels["valid"].reshape(-1,10),predictions=prediction)
    merged={**base["config"],**config}
    result={"name":config["name"],"cutoff":base["cutoff"],"method":"catboost_event",**variants[(temperature,mix)],
            "config":merged,"feature_dataset":base["feature_dataset"],"tree_count":base["tree_count"],
            "event_tree_count":gate.tree_count_,"temperature":temperature,"event_mix":mix,
            "seconds":time.monotonic()-start,"base_auc":base["auc"],"base_experiment":base["name"],
            "all_variants":{f"T{t}_mix{m}":r for (t,m),r in variants.items()}}
    write_json(out/"result.json",result);(out/"config.json").write_text(json.dumps(merged,indent=2))
    note(f"事件分解实验 {result['name']}：AUC {result['auc']:.6f}，基础模型 {base['auc']:.6f}，温度 {temperature}，事件分支占比 {mix}。保留未点击结果。","experiment")
    print(json.dumps(result,indent=2),flush=True)
