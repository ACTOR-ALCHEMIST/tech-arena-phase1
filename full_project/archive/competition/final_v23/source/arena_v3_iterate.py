"""Run five declared candidates, locked second-cutoff audits, then best export."""
import argparse,fcntl,json,os,subprocess,sys,time,itertools
from pathlib import Path
import numpy as np
from arena_v3 import ROOT,OUT,RECIPES,guard,sha
from arena_report import write_json

EXTRAS=['v3_wide512_s73','v3_cross512_s2027']


def command(args,label):
    write_json(OUT/'state.json',dict(stage=label,command=args,started=time.time()))
    log=OUT/(label+'.log')
    with log.open('a') as f:
        subprocess.run([sys.executable,'-u']+args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def candidates():return list(RECIPES)+EXTRAS


def record(name,cutoff):
    if name in RECIPES:
        path=OUT/'runs'/name/cutoff;r=json.loads((path/'result.json').read_text())
    else:
        path=ROOT/'competition/experiments'/(name if cutoff=='2025-01-26' else name+'_locked'+cutoff[-2:]);r=json.loads((path/'result.json').read_text())
    d=np.load(path/'validation.npz');return r,d


def select():
    from arena_finalize import blend
    from arena_core import evaluate
    from arena_validation import importance_weights,weighted_auc
    names=candidates();groups={};weights=[[1.]+[0.]*len(names)]
    for i in range(len(names)):
        for w in [.05,.1,.25,.5,1.]:
            row=[1-w]+[0.]*len(names);row[i+1]=w;weights.append(row)
    for a,b in itertools.combinations(range(len(names)),2):
        for mass in [.25,.5]:
            row=[1-mass]+[0.]*len(names);row[a+1]=row[b+1]=mass/2;weights.append(row)
    for mass in [.25,.5]:weights.append([1-mass]+[mass/len(names)]*len(names))
    for cutoff in ['2025-01-26','2025-01-25']:
        folder=OUT/'data'/cutoff;y=np.load(folder/'valid_labels.npy');indices=np.load(folder/'valid_indices.npy')
        prediction=[np.load(folder/'incumbent.npy')]
        for name in names:
            _,d=record(name,cutoff);np.testing.assert_array_equal(d['event_indices'],indices);np.testing.assert_array_equal(d['labels'],y)
            prediction.append(d['predictions'])
        iw,_=importance_weights(cutoff,indices);groups[cutoff]=(y,prediction,iw)
    scores=[]
    for w in weights:
        row=dict(weights=w,kind='logit')
        for cutoff,(y,pred,iw) in groups.items():
            p=blend(pred,w,'logit');row['auc_'+cutoff[-2:]]=evaluate(y,p)['auc'];row['weighted_auc_'+cutoff[-2:]]=weighted_auc(y,p,iw)
        row['selection_score']=(row['weighted_auc_26']+row['weighted_auc_25'])/2;scores.append(row)
    old=scores[0]
    eligible=[r for r in scores[1:] if all(r[k]>old[k]+1e-6 for k in ['auc_26','auc_25','selection_score'])]
    chosen=max(eligible,key=lambda r:r['selection_score']) if eligible else old
    result=dict(sources=['frozen_v2']+names,selected=chosen,incumbent=old,eligible=len(eligible),recipes=scores,
        candidates=[dict(name=n,primary=record(n,'2025-01-26')[0],secondary=record(n,'2025-01-25')[0]) for n in names],
        protocol='Declared five-model pool and coarse logit mixtures; select only recipes improving raw AUC on both Jan26/Jan25 and mean cohort-weighted AUC. Audit epochs/schedules locked to Jan26 selection. No official evaluation labels.')
    write_json(OUT/'selection.json',result)
    for cutoff,(y,pred,iw) in groups.items():
        p=blend(pred,chosen['weights'],'logit')
        np.savez_compressed(OUT/('selected_validation_'+cutoff[-2:]+'.npz'),event_indices=np.load(OUT/'data'/cutoff/'valid_indices.npy'),labels=y,predictions=p)
    print(json.dumps(dict(stage='selection',selected=chosen,incumbent=old,eligible=len(eligible))),flush=True)
    return result


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.iteration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initial=guard();write_json(OUT/'incumbent.json',initial)
        command(['arena_v3.py','train','--name','r1_history128'],'r1_primary')
        best=record('r1_history128','2025-01-26')[0]['best_epoch']
        command(['arena_v3.py','train','--name','r1_history128','--cutoff','2025-01-25','--fixed-epochs',str(best)],'r1_locked25')
        for name in EXTRAS:
            command(['arena_tabular.py','--config','configs/'+name+'.json'],name+'_primary')
            r=json.loads((ROOT/'competition/experiments'/name/'result.json').read_text())
            command(['-c',"import json;from arena_finalize import ensure_secondary;from pathlib import Path;ensure_secondary(json.loads(Path("+repr(str(ROOT/'competition/experiments'/name/'result.json'))+").read_text()))"],name+'_locked25')
        for name in ['r2_qwen3_history128','r3_qwen25_history192']:
            foundation=RECIPES[name]['foundation'];source=OUT/'model_sources'/(foundation+'.json')
            started=time.monotonic()
            while not source.exists():
                write_json(OUT/'state.json',dict(stage='waiting_foundation',name=foundation,seconds=time.monotonic()-started))
                if time.monotonic()-started>3600:raise TimeoutError('Foundation download did not complete; inspect v3_download.log')
                time.sleep(10)
            for cutoff in ['2025-01-26','2025-01-25']:
                command(['arena_v3_foundation.py','--name',foundation,'--cutoff',cutoff,'--layers',str(RECIPES[name].get('foundation_layers',12))],name+'_features'+cutoff[-2:])
                args=['arena_v3.py','train','--name',name,'--cutoff',cutoff]
                if cutoff=='2025-01-25':args+=['--fixed-epochs',str(record(name,'2025-01-26')[0]['best_epoch'])]
                command(args,name+'_fit'+cutoff[-2:])
        assert guard()==initial
        select();command(['arena_v3_export.py','export'],'final_export')
        command(['arena_v3_export.py','verify'],'final_verify')
        assert guard()==initial
        write_json(OUT/'state.json',dict(stage='completed',finished=time.time()))


if __name__=='__main__':run()
