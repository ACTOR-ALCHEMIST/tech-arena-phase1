"""Frozen two-stage refit, separate-process raw replay, and CSV promotion."""
import os
import argparse
import copy
import json
import shutil
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F
from arena_ood import ROOT,sha,write_json,guard
from arena_two_stage_data import OUT,prepare,fit_ranker,ranker_recipe,stage1_features,raw_inputs,relation_features
from arena_two_stage import Inputs,Reranker,predict,training_inputs,training_objective
from arena_two_stage_campaign import compose

FINAL=ROOT/'competition/final_v21'
RUNS=OUT/'runs'


def promote_verified():
    manifest=json.loads((FINAL/'manifest.json').read_text())
    assert guard()==manifest['incumbent_digest']
    assert json.loads((FINAL/'acceptance.json').read_text())['passed']
    assert sha(FINAL/'submission.csv')==manifest['sha256']
    write_json(ROOT/'competition/CURRENT_BEST.json',dict(csv=manifest['csv'],sha256=manifest['sha256'],
               manifest=str((FINAL/'manifest.json').relative_to(ROOT)),acceptance=str((FINAL/'acceptance.json').relative_to(ROOT)),
               validation_26=manifest['validation_26'],validation_25=manifest['validation_25'],validation_24=manifest['validation_24'],
               previous_best_csv=manifest['previous_best']['csv'],scope='Best validated local recipe; official evaluation labels unavailable.'))


def resume_verification():
    manifest=json.loads((FINAL/'manifest.json').read_text())
    assert guard()==manifest['incumbent_digest']
    subprocess.run([sys.executable,'-u',str(FINAL/'source/arena_two_stage_export.py'),'--verify','--final',str(FINAL)],cwd=ROOT,check=True)
    promote_verified()


def load_reranker(path,device):
    c=torch.load(path,map_location='cpu',weights_only=False);s=c['state_dict']
    m=Reranker(s['mean'].numpy(),s['scale'].numpy(),c['config']);m.load_state_dict(s)
    return m.to(device).eval()


def load_first(path,device):
    from arena_push_model import build_model
    c=torch.load(path,map_location='cpu',weights_only=False);s=c['state_dict']
    m=build_model(c['width'],s['mean'].numpy(),s['scale'].numpy(),c['config']);m.load_state_dict(s)
    return m.to(device).eval()


def evaluation_inputs(raw,stage1_path,fc,device):
    from arena_tabular import predict as first_predict
    from arena_spherical_repr import extra_features,model as sphere_model
    from arena_daily import StackRows
    ix=np.arange(len(raw['users']));inputs=raw_inputs(raw)
    x,names=stage1_features(inputs,ix,'2025-01-27',fc)
    p=first_predict(load_first(stage1_path,device),x,device)
    sphere,cols=extra_features(inputs,ix,'2025-01-27',fc)
    values=np.concatenate([x,sphere],1);del x,sphere
    rep=dict(np.load(sphere_model('2025-01-27')/'factors_e32.npz'))
    data=Inputs.__new__(Inputs);data.features=StackRows(values);data.indices=ix;data.prob=p
    data.pairs=relation_features(rep,inputs,ix)
    return data,names+cols


def fit_final(name,path,epochs):
    from arena_push_model import normalizer
    from arena_objectives import loss as objective
    from arena_two_stage import CONFIGS
    if (path/'model.pt').exists():return
    path.mkdir(parents=True,exist_ok=True)
    config=copy.deepcopy(CONFIGS[name])
    data,training_days=training_inputs('2025-01-27',config)
    mean,scale=normalizer(data.features,{'normalizer':'relative_range'})
    torch.manual_seed(config['seed']);torch.set_num_threads(2)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=Reranker(mean,scale,config).to(device);ema=copy.deepcopy(model).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.03)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=.00005)
    rng=np.random.default_rng(config['seed']);records=[]
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(data.indices));model.train();total=0.
        for start in range(0,len(order),1024):
            take=order[start:start+1024];x,p,pair=data.batch(take,device)
            y=torch.tensor(data.labels[take],dtype=torch.float32,device=device);z=model(x,p,pair)
            loss=training_objective(z,y,config,p,pair,x[:,0,25])
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
            total+=loss.item()*len(take)
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.mul_(.99).add_(b,alpha=.01)
        scheduler.step();records.append(dict(epoch=epoch,loss=total/len(order)))
        print(json.dumps(dict(stage='final_reranker_fit',name=name,**records[-1])),flush=True)
    torch.save(dict(state_dict=ema.cpu().state_dict(),config=config,width=len(mean)),path/'model.pt')
    write_json(path/'training.json',records)
    write_json(path/'recipe.json',dict(config=config,epochs=epochs,training_days=training_days,
               events=len(data.indices),policy='Chronological OOF score inputs. Epoch count locked on primary validation.'))


def export():
    from ctr_data import load_cache
    from arena_ood_export import csv_probabilities
    from arena_finalize import export_csv
    from arena_core import snapshot
    if (FINAL/'manifest.json').exists():
        resume_verification();return
    selection=json.loads((OUT/'selection.json').read_text());chosen=selection['selected']
    assert chosen and json.loads((OUT/'promotion.json').read_text())['passed']
    before=guard();assert before==json.loads((OUT/'incumbent.json').read_text())
    FINAL.mkdir(parents=True,exist_ok=True)
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text())
    shutil.copy2(ROOT/pointer['csv'],FINAL/'frozen_incumbent.csv')
    name=chosen['name'];primary=json.loads((RUNS/name/'2025-01-26/result.json').read_text())
    fit_final(name,FINAL/'models/reranker',primary['best_epoch'])
    ranker=fit_ranker('2025-01-27');(FINAL/'models/ranker').mkdir(parents=True,exist_ok=True)
    for p in ['model.pt','result.json']:shutil.copy2(ranker/p,FINAL/'models/ranker'/p)
    _,raw=load_cache(ROOT/'data/cache_v1','evaluation')
    cfg,fc=ranker_recipe('2025-01-27');device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    inputs,names=evaluation_inputs(raw,FINAL/'models/ranker/model.pt',fc,device)
    reranked=predict(load_reranker(FINAL/'models/reranker/model.pt',device),inputs,device)
    incumbent=csv_probabilities(FINAL/'frozen_incumbent.csv',raw['rowids'].ravel())
    p=compose(incumbent,inputs.prob,reranked,chosen['mode'],chosen['mass'])
    np.save(FINAL/'stage1_predictions.npy',inputs.prob);np.save(FINAL/'reranked_predictions.npy',reranked)
    np.save(FINAL/'predictions.npy',p);export_csv(FINAL/'submission.csv',raw,p)
    for day in ['26','25','24']:
        shutil.copy2(OUT/('selected_validation_'+day+'.npz'),FINAL/('validation_'+day+'.npz'))
    shutil.copy2(OUT/'audit_24.json',FINAL/'audit_24.json')
    shutil.copy2(OUT/'selection.json',FINAL/'selection.json')
    shutil.copytree(snapshot('2025-01-27'),FINAL/'representations/snapshot_2025-01-27',dirs_exist_ok=True)
    source=FINAL/'source';source.mkdir(exist_ok=True)
    files=list(ROOT.glob('arena_*.py'))+[ROOT/f for f in ['ctr_data.py','ctr_model.py','run.py','test_arena_two_stage.py']]
    for f in files:shutil.copy2(f,source/f.name)
    manifest=dict(method='chronological_fine_ranker_then_bounded_slate_reranker',previous_best=pointer,
                  target_auc=.838,official_evaluation_labels_available=False,selection=chosen,feature_config=fc,feature_names=names,
                  rows=len(p)*10,csv=str((FINAL/'submission.csv').relative_to(ROOT)),sha256=sha(FINAL/'submission.csv'),
                  validation_26=dict(auc=chosen['auc_26']),validation_25=dict(auc=chosen['auc_25']),
                  validation_24=dict(auc=json.loads((OUT/'audit_24.json').read_text())['new_auc']),
                  incumbent_digest=before,
                  artifact_sha256={str(f.relative_to(FINAL)):sha(f) for f in FINAL.rglob('*') if f.is_file() and not f.name.endswith('.lock') and f.name not in ['manifest.json','acceptance.json']},
                  raw_input_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'data/cache_v1/evaluation').glob('*.npy')})
    write_json(FINAL/'manifest.json',manifest);assert guard()==before
    subprocess.run([sys.executable,'-u',str(source/'arena_two_stage_export.py'),'--verify','--final',str(FINAL)],cwd=ROOT,check=True)
    assert guard()==before
    acceptance=json.loads((FINAL/'acceptance.json').read_text());assert acceptance['passed']
    promote_verified()


def verify():
    from ctr_data import load_cache
    from arena_ood_export import csv_probabilities
    from run import check_submission
    workspace=FINAL.resolve().parents[1]
    manifest=json.loads((FINAL/'manifest.json').read_text())
    for f,digest in manifest['artifact_sha256'].items():assert sha(FINAL/f)==digest,f
    for f,digest in manifest['raw_input_sha256'].items():assert sha(workspace/f)==digest,f
    for f,digest in manifest['incumbent_digest'].items():assert sha(workspace/f)==digest,f
    os.environ['ARENA_SNAPSHOT_ROOT']=str(FINAL/'representations')
    torch.set_num_threads(2);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    _,raw=load_cache(workspace/'data/cache_v1','evaluation')
    inputs,names=evaluation_inputs(raw,FINAL/'models/ranker/model.pt',manifest['feature_config'],device)
    assert names==manifest['feature_names']
    pred=predict(load_reranker(FINAL/'models/reranker/model.pt',device),inputs,device)
    incumbent=csv_probabilities(FINAL/'frozen_incumbent.csv',raw['rowids'].ravel())
    chosen=manifest['selection'];full=compose(incumbent,inputs.prob,pred,chosen['mode'],chosen['mass'])
    first_error=float(np.max(np.abs(inputs.prob-np.load(FINAL/'stage1_predictions.npy'))))
    second_error=float(np.max(np.abs(pred-np.load(FINAL/'reranked_predictions.npy'))))
    expected=np.load(FINAL/'predictions.npy')
    raw_full_error=float(np.max(np.abs(full-expected)))
    cached=compose(incumbent,np.load(FINAL/'stage1_predictions.npy'),np.load(FINAL/'reranked_predictions.npy'),chosen['mode'],chosen['mass'])
    composition_error=float(np.max(np.abs(cached-expected)))
    csv=csv_probabilities(FINAL/'submission.csv',raw['rowids'].ravel())
    rounding_error=float(np.max(np.abs(expected-csv)))
    check_submission(FINAL/'submission.csv',workspace/'evaluation.zip')
    acceptance=dict(passed=first_error<2e-5 and second_error<2e-5 and raw_full_error<2e-5 and composition_error<1e-12 and rounding_error<1e-9,
                    rows=int(full.size),stage1_raw_replay_max_abs_error=first_error,reranker_raw_replay_max_abs_error=second_error,
                    full_raw_replay_max_abs_error=raw_full_error,composition_max_abs_error=composition_error,csv_max_abs_error=rounding_error,
                    protocol='Separate process using frozen source, frozen representations and original evaluation inputs; every row checked.')
    write_json(FINAL/'acceptance.json',acceptance);print(json.dumps(acceptance),flush=True)
    assert acceptance['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--final');p.add_argument('--campaign');a=p.parse_args()
    if a.final:FINAL=__import__('pathlib').Path(a.final).resolve()
    if a.campaign:OUT=__import__('pathlib').Path(a.campaign).resolve()
    verify() if a.verify else export()
