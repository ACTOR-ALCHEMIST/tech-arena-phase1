"""Retrieve research + experiment memory and generate the next local experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from arena_report import LAB, note

RESEARCH = [
 {"title":"Orthogonal Procrustes", "url":"https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html",
  "text":"Orthogonal Procrustes aligns coordinate matrices by rotations and reflections without translation or scaling. Orthogonal transformations preserve inner products. Our empirical adaptation aligns independently trained time snapshots to a strictly past anchor before exposing coordinates to a context-aware CTR head; ranking improvement must be validated."},
 {"title":"Implicit ALS", "url":"https://benfred.github.io/implicit/api/models/cpu/als.html",
  "text":"implicit ALS alternating least squares positive confidence alpha factors regularization; warmup history user item matrix; refit cold user factors using provided click history. Avoid treating missing items as observed nonclick labels in supervised CTR."},
 {"title":"EASE", "url":"https://arxiv.org/abs/1905.03375",
  "text":"EASE shallow linear recommendation can outperform deeper recommenders on sparse implicit data. Dense item item inverse requires quadratic memory. With 50000 items and 16GB RAM prefer sparse cooccurrence or low rank ALS rather than full dense inverse."},
 {"title":"Two tower sampling bias", "url":"https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/",
  "text":"Two tower retrieval with in batch negatives has sampling bias. This competition provides actual exposed nonclicks; BCE using observed candidates is the primary click task. Auxiliary history item representation pretraining can use explicit sampled objectives."},
 {"title":"CatBoost", "url":"https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic",
  "text":"CatBoost transforms categorical features using target statistics. Greedy target statistics leak the current label. Use time safe feature snapshots; combine co click scores latent ALS interaction match and observed context. Use Logloss training with validation AUC and early stopping."},
 {"title":"DIN", "url":"https://arxiv.org/abs/1706.06978",
  "text":"DIN candidate aware attention represents diverse user interests from click history. Time decay and repeated click features complement long term user embeddings. Prefer small model and regularization when sparse IDs overfit quickly."},
 {"title":"Temporal evaluation", "url":"https://arxiv.org/abs/2010.11060",
  "text":"Respect global temporal order; leakage can change algorithm rankings. Use multiple global cutoffs and one first event per user. Generate training target features strictly from earlier snapshots; evaluation labels are not available."},
 {"title":"Item2Vec", "url":"https://arxiv.org/abs/1603.04259",
  "text":"Item2Vec uses skip gram negative sampling for item similarity in implicit feedback recommendation. History based auxiliary embedding learning complements exposed click classification. Our user item SGNS experiment is a distinct MF objective inspired by sampled representation learning."},
 {"title":"LightGCN", "url":"https://arxiv.org/abs/2002.02126",
  "text":"LightGCN retains linear neighborhood aggregation and layer combination, omitting unnecessary feature transformations and nonlinear activation. Sparse user item graph propagation and low rank regularization are promising under limited memory."}
]


def retrieve(query, k=4):
    documents=list(RESEARCH)
    for p in sorted((LAB/"experiments").glob("*/result.json")):
        r=json.loads(p.read_text())
        documents.append({"title":r["name"],"url":str(p),
                          "text":f"{r['method']} AUC {r['auc']:.6f} cutoff {r['cutoff']} config {json.dumps(r.get('config',{}))} features {r.get('feature_importance_top10',[])}"})
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=10000)
    x=vec.fit_transform([d["text"] for d in documents]+[query])
    similarities=cosine_similarity(x[-1],x[:-1]).ravel()
    chosen=np.argsort(-similarities)[:k]
    return [{**documents[i],"similarity":float(similarities[i])} for i in chosen]


def candidates(lane="cpu"):
    if lane=="gpu":
        configs=[{"name":f"sgns{dim}_n{neg}_e80_s{seed}","method":"sgns","factors":dim,"device":"mps",
                 "epochs":80,"batch_size":8192,"lr":0.02,"negatives":neg,"regularization":reg,
                 "lr_decay":True,"patience":8,"seed":seed}
                for dim,neg,reg,seed in [(16,8,0.001,42),(32,8,0.001,42),(8,8,0.001,42),(4,8,0.001,42),
                                          (6,8,0.001,42),(12,8,0.001,42),(8,16,0.001,42),(8,8,0.001,2026),
                                          (16,16,0.001,42),(16,8,0.001,2026),(32,8,0.003,2026)]]
        configs.append({"name":"sgns8_distinct_e80_s42","method":"sgns","factors":8,"device":"mps",
                        "epochs":80,"batch_size":8192,"lr":0.02,"negatives":8,"regularization":0.001,
                        "lr_decay":True,"patience":8,"seed":42,"feedback_mode":"distinct_clicks"})
        configs.append({"name":"sgns8_const_e160_s42","method":"sgns","factors":8,"device":"mps",
                        "epochs":160,"batch_size":8192,"lr":0.01,"negatives":8,"regularization":0.001,
                        "lr_decay":False,"patience":12,"seed":42})
        for dim,neg,mode,epochs in [(8,8,"binary",320),(8,4,"binary",160),(8,8,"distinct_clicks",160),
                                    (6,8,"binary",160),(12,8,"binary",160),(8,8,"binary",640)]:
            configs.append({"name":f"sgns{dim}_const_n{neg}_{mode}_e{epochs}_s42","method":"sgns",
                "factors":dim,"device":"mps","epochs":epochs,"batch_size":8192,"lr":0.01,"negatives":neg,
                "regularization":0.001,"lr_decay":False,"patience":8,"seed":42,"feedback_mode":mode})
        for c in configs:c["eval_every"]=5
        configs.append({"name":"sgns8_const_e320_s2026","method":"sgns","factors":8,"device":"mps",
                        "epochs":320,"batch_size":8192,"lr":0.01,"negatives":8,"regularization":0.001,
                        "lr_decay":False,"eval_every":10,"patience":8,"seed":2026})
        configs.sort(key=lambda c:abs(c["factors"]-8))
        for source in ("cat_sgns8_16_context_24_180000_d6","cat_sgns8long_context_24_180000_d6"):
            if (LAB/"experiments"/source/"result.json").exists():
                for loss in ("event_ce","bce"):
                    configs.append({"name":f"mlp_{loss}_{source}","method":"tabular","source_experiment":source,
                                    "loss":loss,"epochs":30,"lr":0.001,"dropout":0.1,"weight_decay":0.01})
        for source in ("cat_sgns8e320_context24_d6","cat_foldin_long24_d6","cat_foldin320_context24_d6","cat_sgns8_s2026_context24_d6","cat_sgns8e640_context24_d6"):
            if (LAB/"experiments"/source/"result.json").exists():
                configs.append({"name":"mlp_event_"+source,"method":"tabular","source_experiment":source,
                                "loss":"event_ce","epochs":40,"lr":0.001,"dropout":0.1,"weight_decay":0.01,
                                "ema":0.99,"seed":2026,"constant_policy":"zero"})
        for source in ("cat_sgns8long_context_24_180000_d6","cat_sgns8e320_context24_d6"):
            if (LAB/"experiments"/source/"result.json").exists():
                configs.append({"name":"mlp_context_"+source,"method":"tabular","source_experiment":source,
                                "loss":"event_ce","epochs":40,"lr":0.001,"dropout":0.1,"weight_decay":0.01,
                                "architecture":"context","ema":0.99,"seed":2026,"constant_policy":"zero"})
        if (LAB/'experiments/cat_sgns8_s2026_context24_d6/result.json').exists():
            configs.append({'name':'mlp_aligned_s2026_e320','method':'tabular','source_experiment':'cat_sgns8_s2026_context24_d6',
                            'loss':'event_ce','epochs':40,'lr':0.001,'dropout':0.1,'weight_decay':0.01,'ema':0.99,
                            'seed':2026,'constant_policy':'zero','aligned_coordinates':True})
        source="cat_sgns8long_context_24_180000_d6"
        if (LAB/"experiments"/source/"result.json").exists():
            for seed,dropout,decay in [(2026,0.1,0.99),(7,0.2,0.99),(42,0.2,0.99)]:
                configs.append({"name":f"mlp_event_long_ema_s{seed}_d{dropout}","method":"tabular","source_experiment":source,
                                "loss":"event_ce","epochs":40,"lr":0.001,"dropout":dropout,"weight_decay":0.01,"ema":decay,"seed":seed,"constant_policy":"zero"})
            configs.append({"name":"mlp_event_long_constzero","method":"tabular","source_experiment":source,
                            "loss":"event_ce","epochs":30,"lr":0.001,"dropout":0.1,"weight_decay":0.01,"constant_policy":"zero"})
        audit_base=source+"_locked25";reference="mlp_event_ce_"+source
        if (LAB/"experiments"/audit_base/"result.json").exists() and (LAB/"experiments"/reference/"result.json").exists():
            r=json.loads((LAB/"experiments"/reference/"result.json").read_text());c=dict(r["config"])
            c.update(name=reference+"_locked25",source_experiment=audit_base,epochs=r["best_epoch"],fixed_epochs=True,
                     lr_schedule_epochs=r["config"].get("epochs",30),audit_protocol="locked rolling origin")
            configs.append(c)
        leaders=[json.loads(p.read_text()) for p in (LAB/'experiments').glob('*/result.json')]
        leaders=sorted([r for r in leaders if r['method']=='tabular_mlp_mps' and r['cutoff']=='2025-01-26'],key=lambda r:-r['auc'])[:3]
        for r in leaders:
            audit_base=r['source_experiment']+'_locked25'
            if (LAB/'experiments'/audit_base/'result.json').exists():
                c=dict(r['config']);c.update(name=r['name']+'_locked25',source_experiment=audit_base,
                    epochs=r['best_epoch'],fixed_epochs=True,lr_schedule_epochs=r['config'].get('epochs',30),audit_protocol='locked rolling origin')
                configs.append(c)
        return configs
    plans=[]
    for factors,alpha,reg in [(32,10,0.1),(16,10,0.1),(16,3,0.1),(8,10,0.1),(24,5,0.1),
                              (128,10,0.1),(128,40,0.5),(256,20,0.5),(64,5,0.1),(128,5,0.1)]:
        plans.append({"name":f"als{factors}_a{alpha}_r{reg}","method":"als","factors":factors,"alpha":alpha,"regularization":reg,"iterations":25})
    for events,depth,l2 in [(80000,4,20),(200000,6,20),(200000,8,30)]:
        plans.append({"name":f"cat{events}_d{depth}_l{l2}","method":"catboost","max_events":events,"depth":depth,"l2":l2,"iterations":1000,"lr":0.05})
    for events,depth in [(80000,6),(200000,6),(200000,8)]:
        plans.append({"name":f"cat_als16_32_{events}_d{depth}","method":"catboost","max_events":events,
                      "depth":depth,"l2":20,"iterations":1200,"lr":0.05,"latent_factors":[16,32]})
    for start,events,depth in [("2025-01-22",200000,6),("2025-01-24",180000,6),("2025-01-24",180000,8)]:
        plans.append({"name":f"cat_context_{start[-2:]}_{events}_d{depth}","method":"catboost","max_events":events,
                      "depth":depth,"l2":20,"iterations":1200,"lr":0.05,"latent_factors":[16,32],
                      "train_start":start,"context_features":True})
    for start,events,depth in [("2025-01-22",80000,6),("2025-01-24",180000,6),("2025-01-24",180000,8)]:
        plans.append({"name":f"cat_sgns_context_{start[-2:]}_{events}_d{depth}","method":"catboost","max_events":events,
                      "depth":depth,"l2":20,"iterations":1500,"lr":0.04,"latent_factors":[16,32],
                      "train_start":start,"context_features":True,"sgns_factors":[16]})
        plans.append({"name":f"cat_sgns8_16_context_{start[-2:]}_{events}_d{depth}","method":"catboost","max_events":events,
                      "depth":depth,"l2":20,"iterations":1500,"lr":0.04,"latent_factors":[8,16],
                      "train_start":start,"context_features":True,"sgns_factors":[8,16]})
    for base in ("cat_sgns8_16_context_22_80000_d6","cat_sgns8_16_context_24_180000_d6",
                 "cat_sgns8_16_context_24_180000_d8","cat_sgns8long_context_24_180000_d6","cat_sgns8long_context_24_180000_d8",
                 "cat_rolling24_sgns8_16_d6","cat_rolling24_sgns8_16_d8","cat_sgns8e320_context24_d6",
                 "cat_sgns8_s2026_context24_d6","cat_sgns8e640_context24_d6"):
        if (LAB/"experiments"/base/"result.json").exists():
            plans.append({"name":"event_"+base,"method":"event","source_experiment":base,"iterations":1200})
    audited_base="cat_sgns8_16_context_24_180000_d6_locked25"
    primary_event="event_cat_sgns8_16_context_24_180000_d6"
    if (LAB/"experiments"/audited_base/"result.json").exists() and (LAB/"experiments"/primary_event/"result.json").exists():
        reference=json.loads((LAB/"experiments"/primary_event/"result.json").read_text())
        plans.append({"name":primary_event+"_locked25","method":"event","source_experiment":audited_base,
                      "iterations":reference["event_tree_count"],"fixed_trees":True,
                      "locked_temperature":reference["temperature"],"locked_mix":reference["event_mix"]})
    raw_source="sgns8_const_n8_binary_e320_s42"
    if (LAB/"experiments"/raw_source/"result.json").exists():
        raw=json.loads((LAB/"experiments"/raw_source/"result.json").read_text())
        plans.append({"name":"cat_sgns8e320_context24_d6","method":"catboost","max_events":180000,
            "depth":6,"l2":20,"iterations":2000,"lr":0.04,"latent_factors":[8,16],"train_start":"2025-01-24",
            "context_features":True,"sgns_factors":[8,16],"sgns_epochs":{"8":raw["best_epoch"],"16":56},
            "sgns_recipes":{"8":{"epochs":raw["best_epoch"],"lr":0.01,"lr_decay":False,"reuse_experiment":raw_source}}})
    for raw_source,name in [('sgns8_const_e320_s2026','cat_sgns8_s2026_context24_d6'),
                            ('sgns8_const_n8_binary_e640_s42','cat_sgns8e640_context24_d6')]:
        if (LAB/'experiments'/raw_source/'result.json').exists():
            raw=json.loads((LAB/'experiments'/raw_source/'result.json').read_text())
            if raw['auc']>0.795:
                plans.append({'name':name,'method':'catboost','max_events':180000,'depth':6,'l2':20,'iterations':2000,
                    'lr':0.04,'latent_factors':[8,16],'train_start':'2025-01-24','context_features':True,
                    'sgns_factors':[8,16],'sgns_epochs':{'8':raw['best_epoch'],'16':56},
                    'sgns_recipes':{'8':{'epochs':raw['best_epoch'],'lr':0.01,'lr_decay':False,'seed':raw['config'].get('seed',42),'reuse_experiment':raw_source}}})
    base="cat_sgns8long_context_24_180000_d6"
    if (LAB/"experiments"/base/"result.json").exists():
        r=json.loads((LAB/"experiments"/base/"result.json").read_text());c=dict(r["config"])
        c.update(name=base+"_locked25",cutoff="2025-01-25",train_start="2025-01-23",iterations=r["tree_count"],
                 fixed_trees=True,audit_protocol="locked rolling origin")
        plans.append(c)
    plans.append({"name":"cat_sgns8e80_16_context_22_80000_d6","method":"catboost","max_events":80000,
                  "depth":6,"l2":20,"iterations":2000,"lr":0.04,"latent_factors":[8,16],"train_start":"2025-01-22",
                  "context_features":True,"sgns_factors":[8,16],"sgns_epochs":{"8":80,"16":56}})
    for start,events,depth in [("2025-01-24",180000,6),("2025-01-24",180000,8)]:
        plans.append({"name":f"cat_sgns8long_context_{start[-2:]}_{events}_d{depth}","method":"catboost","max_events":events,
            "depth":depth,"l2":20,"iterations":2000,"lr":0.04,"latent_factors":[8,16],"train_start":start,
            "context_features":True,"sgns_factors":[8,16],"sgns_epochs":{"8":159,"16":56},
            "sgns_recipes":{"8":{"epochs":159,"lr":0.01,"lr_decay":False,"reuse_experiment":"sgns8_const_e160_s42"}}})
    for depth in (6,8):
        plans.append({"name":f"cat_rolling24_sgns8_16_d{depth}","method":"catboost","max_events":180000,
            "depth":depth,"l2":20,"iterations":2000,"lr":0.04,"latent_factors":[8,16],"train_start":"2025-01-24",
            "context_features":True,"sgns_factors":[8,16],"sgns_epochs":{"8":80,"16":56},"rolling":True})
    plans.append({"name":"cat_foldin_long24_d6","method":"catboost","max_events":180000,
        "depth":6,"l2":20,"iterations":2000,"lr":0.04,"latent_factors":[8,16],"train_start":"2025-01-24",
        "context_features":True,"sgns_factors":[8,16],"sgns_epochs":{"8":159,"16":56},
        "sgns_recipes":{"8":{"epochs":159,"lr":0.01,"lr_decay":False,"reuse_experiment":"sgns8_const_e160_s42"}},
        "foldin":{"factors":8,"steps":30,"lr":0.1,"anchor":0.01,"negatives":512}})
    folded=LAB/'experiments/cat_foldin_long24_d6/result.json'
    source=LAB/'experiments/cat_sgns8e320_context24_d6/result.json'
    if folded.exists() and source.exists():
        f=json.loads(folded.read_text())
        if f['auc']>0.8200:
            c=dict(json.loads(source.read_text())['config']);c.update(name='cat_foldin320_context24_d6',foldin=f['config']['foldin'])
            plans.append(c)
    for name in ('cat_sgns8e320_context24_d6','cat_sgns8_s2026_context24_d6','cat_sgns8e640_context24_d6','cat_foldin_long24_d6','cat_foldin320_context24_d6'):
        p=LAB/'experiments'/name/'result.json'
        if p.exists():
            r=json.loads(p.read_text())
            if r['auc']>0.82:
                c=dict(r['config']);c.update(name=name+'_locked25',cutoff='2025-01-25',train_start='2025-01-23',
                    iterations=r['tree_count'],fixed_trees=True,audit_protocol='locked rolling origin')
                plans.append(c)
    return plans


def next_plan(lane="cpu"):
    research=LAB/"research";research.mkdir(parents=True,exist_ok=True)
    (research/"sources.json").write_text(json.dumps(RESEARCH,indent=2,ensure_ascii=False))
    results=[json.loads(p.read_text()) for p in (LAB/"experiments").glob("*/result.json")]
    best_als=max((r["auc"] for r in results if r["method"]=="implicit_als" and r["cutoff"]=="2025-01-26"),default=0)
    best_cat=max((r["auc"] for r in results if r["method"].startswith("catboost") and r["cutoff"]=="2025-01-26"),default=0)
    pool=candidates(lane)
    # Adaptive allocation: prioritize a promising family over the weaker family.
    pool.sort(key=lambda c: (-5 if c.get("audit_protocol") else -4 if c["method"]=="tabular" else -3 if c["method"]=="event" else -2 if c.get("sgns_factors") else -1 if c.get("latent_factors") and best_als>best_cat+0.01 else
                            0 if (c["method"]=="als" and best_als>best_cat+0.02) else 1,
                            c.get("max_events",0)))
    for config in pool:
        out=LAB/"experiments"/config["name"]
        if (out/"result.json").exists() or (out/"failure.json").exists() or (out/'pruned.json').exists():continue
        reason=None
        if config['method']=='sgns' and config.get('factors',8)>=16 and config.get('seed',42)!=42:
            if any(r['method'].startswith('sgns_') and r.get('config',{}).get('factors')==8 and r['auc']>0.8 for r in results):
                reason='8-dimensional representation exceeded AUC 0.80; completed 16/32-dimensional runs were materially weaker. Reallocate seed budget to the stronger low-dimensional family.'
        if config['method']=='catboost' and not config.get('sgns_factors') and best_cat>0.8:
            reason='Numeric/ALS-only heads have completed representatives and trail SGNS-enhanced heads by over 0.02 AUC; prioritize stronger representations and locked audits.'
        if reason:
            out.mkdir(parents=True,exist_ok=True);(out/'pruned.json').write_text(json.dumps({'config':config,'reason':reason},indent=2));note(config['name']+': '+reason,'pruned');continue
        query=("warmup history latent factors alpha regularization ALS confidence improve validation" if config["method"]=="als" else
               "auxiliary implicit feedback matrix factorization sampled negatives GPU history regularization" if config["method"]=="sgns" else
               "tabular_mlp_mps event_ce no click softmax numerical context sgns low rank EMA regularization" if config['method']=='tabular' else
               "time safe snapshots CatBoost co click latent matching feature interactions regularization")
        query+=' '+json.dumps(config,sort_keys=True)
        if config.get('aligned_coordinates'):query+=' orthogonal Procrustes coordinate alignment temporal snapshots'
        evidence=retrieve(query)
        best_neural=max((r['auc'] for r in results if r['method']=='tabular_mlp_mps' and r['cutoff']=='2025-01-26'),default=0)
        reason=f"Current best ALS AUC={best_als:.6f}; tree-family AUC={best_cat:.6f}; event neural head AUC={best_neural:.6f}. Test {config['method']} with retrieved research and matching configuration history; preserve temporal snapshots and first-event cohorts."
        script={"sgns":"arena_sgns.py","tabular":"arena_tabular.py"}.get(config["method"],"arena_models.py")
        plan={"name":config["name"],"argv":[script,"--config",str(LAB/"plans"/(config["name"]+".json"))],"reason":reason,"retrieved":evidence,"config":config}
        (LAB/"plans").mkdir(parents=True,exist_ok=True)
        (LAB/"plans"/(config["name"]+".json")).write_text(json.dumps(config,indent=2))
        out.mkdir(parents=True,exist_ok=True)
        (out/"retrieval_plan.json").write_text(json.dumps(plan,indent=2))
        return plan
    return {"done":True,"reason":"Current candidate queue exhausted; review results and add evidence-backed methods before finalization."}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["next","retrieve"]);p.add_argument("--lane",choices=["cpu","gpu"],default="cpu");p.add_argument("--query",default="latent CTR history sparse recommendation");a=p.parse_args()
    print(json.dumps(next_plan(a.lane) if a.command=="next" else retrieve(a.query),ensure_ascii=False))
