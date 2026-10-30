"""Past-only slate-choice adaptation of the accepted unit-sphere geometry.

Global coefficients are calibrated against an earlier, frozen geometry. Only
unit-normalized user/item vectors adapt on actual clicked slates afterward.
No user intercept or outside-option parameter can absorb the ranking signal.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from arena_ood import ROOT,write_json,guard,sha
from arena_metric_geometry import cosine,preference_probabilities

OUT=ROOT/'competition/conditional_sphere'
RECIPE=dict(version=1,epochs=32,outputs=[0,8,16,32],lr=.002,minimum_lr=.0002,
            batch_size=4096,seed=83891,ema=.98,calibration_max=100000,
            calibration_iterations=40,prior_scales=[.1,1.,10.],masses=[.25,.5,1.])


def unit(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.linalg.norm(x,axis=-1,keepdims=True).clip(1e-8)


def covariates(raw,ix,rep,ib):
    """Seven candidate covariates, independent of labels and coordinates."""
    u,v=unit(rep['users']),unit(rep['items']);parts=[]
    for start in range(0,len(ix),2048):
        b=ix[start:start+2048];ids=raw['items'][b];hist=raw['history'][b]
        h=v[hist];c=v[ids];visible=hist!=0
        sim=np.einsum('bid,btd->bit',c,h)
        length=visible.sum(1);mean=(sim*visible[:,None]).sum(2)/length.clip(1)[:,None]
        w=np.exp(-raw['age_hours'][b]/72)*visible
        decay=(sim*w[:,None]).sum(2)/w.sum(1).clip(1e-8)[:,None]
        peak=np.where(visible[:,None],sim,-1).max(2);peak[length==0]=0
        repeat=(ids[:,:,None]==hist[:,None])&visible[:,None]
        dot=np.einsum('bd,bid->bi',u[raw['users'][b]],c)
        parts.append(np.stack([dot,ib[ids],mean,decay,peak,repeat.any(2),np.log1p(repeat.sum(2))],2).astype('f'))
    return np.concatenate(parts)


class ConditionalSphere(nn.Module):
    def __init__(self,rep,counts,support,temperature,prior):
        super().__init__();self.prior=prior;self.temperature=float(temperature)
        u,v=unit(rep['users']),unit(rep['items'])
        self.users=nn.Embedding.from_pretrained(torch.tensor(u),freeze=False,padding_idx=0)
        self.items=nn.Embedding.from_pretrained(torch.tensor(v),freeze=False,padding_idx=0)
        for key,x in [('u0',u),('v0',v),('uk',np.linalg.norm(u,axis=1)>0),('vk',np.linalg.norm(v,axis=1)>0),
                      ('up',(np.asarray(support[0])/np.maximum(counts[0],1)).astype('f')),
                      ('vp',(np.asarray(support[1])/np.maximum(counts[1],1)).astype('f'))]:
            self.register_buffer(key,torch.tensor(x))

    def lookup(self,ids,user=False):
        table=self.users if user else self.items;known=self.uk if user else self.vk
        return F.normalize(table(ids),dim=-1,eps=.1)*known[ids][...,None]

    def forward(self,uid,ids,offset):
        return self.temperature*(self.lookup(uid,True)[:,None]*self.lookup(ids)).sum(2)+offset

    def penalty(self,uid,ids):
        a=(self.lookup(uid,True)-self.u0[uid]).square().sum(1)*self.up[uid]
        b=(self.lookup(ids)-self.v0[ids]).square().sum(2)*self.vp[ids]
        return .5*self.prior*(a+b.sum(1)).mean()

    @torch.no_grad()
    def representation(self):
        return dict(users=self.lookup(torch.arange(len(self.u0)),True).numpy(),
                    items=self.lookup(torch.arange(len(self.v0))).numpy())


def calibrate(day,raw,fit):
    from arena_two_stage_data import date_shift
    from arena_latent_delta_export import component
    from arena_core import snapshot
    path=OUT/day/'calibration.json'
    if path.exists():return json.loads(path.read_text())
    prefix=date_shift(day,-2);rep=component(prefix,'unitfused25_delta')[1]
    ib=np.load(snapshot(prefix)/'stats.npz')['ib']
    from run import timestamp
    # The full-prefix variant can fit more slates, but coefficient calibration
    # must remain strictly after its earlier representation anchor.
    chosen=fit[raw['timestamps'][fit]>=timestamp(prefix)]
    assert len(chosen) and int(raw['timestamps'][chosen].max())<timestamp(day)
    if len(chosen)>RECIPE['calibration_max']:
        chosen=np.sort(np.random.default_rng(RECIPE['seed']).choice(chosen,RECIPE['calibration_max'],replace=False))
    x=torch.tensor(covariates(raw,chosen,rep,ib));y=torch.tensor(raw['labels'][chosen].argmax(1),dtype=torch.long)
    coef=nn.Parameter(torch.tensor([6.,1.,.5,.5,.3,1.,.2]));initial=coef.detach().clone()
    opt=torch.optim.LBFGS([coef],lr=.5,max_iter=RECIPE['calibration_iterations'],line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=F.cross_entropy((x*coef).sum(2),y)+.005*(coef-initial).square().mean()
        loss.backward();return loss
    opt.step(closure)
    with torch.no_grad():coef[0].clamp_(1,12)
    result=dict(prefix=prefix,coefficients=coef.detach().tolist(),events=len(chosen),
                first_fit_timestamp=int(raw['timestamps'][chosen].min()),last_fit_timestamp=int(raw['timestamps'][chosen].max()),
                policy='All targets after prefix and before cutoff; prefix geometry frozen; no query labels used.')
    write_json(path,result);return result


def training_indices(raw,day):
    from arena_two_stage_data import date_shift
    from run import timestamp
    start='2025-01-20' if RECIPE.get('fit_window')=='full_prefix' else date_shift(day,-2)
    return np.flatnonzero((raw['timestamps']>=timestamp(start))&(raw['timestamps']<timestamp(day))&(raw['labels'].sum(1)>0))


def metrics(labels,p,baseline,iw):
    from arena_validation import weighted_auc
    clicked=labels.sum(1)>0;logs=np.sort(np.log(np.asarray(baseline).clip(1e-8)),axis=1)
    near=(logs[:,-1]-logs[:,-2]<.5)&clicked
    return dict(auc=float(roc_auc_score(labels.ravel(),p.ravel())),weighted_auc=weighted_auc(labels,p,iw),
                near_tie_top1=float((p[near].argmax(1)==labels[near].argmax(1)).mean()))


def run(day,locked=None):
    from ctr_data import load_cache
    from run import timestamp
    from arena_two_stage_data import date_shift
    from arena_latent_delta_export import component
    from arena_core import snapshot
    from scipy import sparse
    from arena_validation import importance_weights
    OUT.mkdir(parents=True,exist_ok=True);before=guard()
    pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    if (OUT/'incumbent_pointer.json').exists():assert json.loads((OUT/'incumbent_pointer.json').read_text())==pointer
    else:write_json(OUT/'incumbent_pointer.json',pointer)
    if (OUT/'recipe.json').exists():assert json.loads((OUT/'recipe.json').read_text())==RECIPE
    else:write_json(OUT/'recipe.json',RECIPE)
    folder=OUT/day;folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():
        assert json.loads((folder/'result.json').read_text()).get('locked_selection')==locked
        return
    torch.set_num_threads(2);_,raw=load_cache(ROOT/'data/cache_v1','train')
    fit=training_indices(raw,day)
    assert len(fit) and int(raw['timestamps'][fit].max())<timestamp(day)
    cal=calibrate(day,raw,fit);coef=np.asarray(cal['coefficients'],dtype='f')
    rep=component(day,'unitfused25_delta')[1];rep={k:unit(rep[k]) for k in ['users','items']}
    ib=np.load(snapshot(day)/'stats.npz')['ib'];x=covariates(raw,fit,rep,ib)
    offset=torch.tensor((x[:,:,1:]*coef[1:]).sum(2));del x
    uid=torch.tensor(np.array(raw['users'][fit]),dtype=torch.long)
    ids=torch.tensor(np.array(raw['items'][fit]),dtype=torch.long)
    target=torch.tensor(raw['labels'][fit].argmax(1),dtype=torch.long)
    counts=(np.bincount(raw['users'][fit],minlength=len(rep['users'])),np.bincount(raw['items'][fit].ravel(),minlength=len(rep['items'])))
    interactions=sparse.load_npz(snapshot(day)/'interactions.npz')
    support=(np.maximum(np.diff(interactions.indptr),1).astype('f'),np.maximum(np.asarray(interactions.sum(0)).ravel(),1).astype('f'))
    final=day=='2025-01-27'
    if final:
        assert locked is not None
        _,query=load_cache(ROOT/'data/cache_v1','evaluation');vi=np.arange(len(query['users']))
        from arena_metric_infer import read_csv
        base=read_csv(ROOT/pointer['csv'],query['rowids'].ravel());labels=None;iw=None
    else:
        query=raw;baseline=np.load((ROOT/pointer['csv']).parent/('validation_'+day[-2:]+'.npz'))
        vi=baseline['event_indices'];assert not np.intersect1d(fit,vi).size
        iw=importance_weights(day,vi)[0];base=baseline['predictions'];labels=baseline['labels']
    initial_cosine=cosine(rep,query,vi)
    records=[];started=time.monotonic()
    priors=[locked['prior']] if locked else RECIPE['prior_scales']
    epochs=locked['epoch'] if locked else RECIPE['epochs']
    outputs=[epochs] if locked else RECIPE['outputs']
    for prior in priors:
        tag='prior'+str(prior);dest=folder/tag;dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():records+=json.loads((dest/'result.json').read_text())['records'];continue
        torch.manual_seed(RECIPE['seed']);net=ConditionalSphere(rep,counts,support,coef[0],prior);ema=copy.deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=RECIPE['lr'],weight_decay=0)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=RECIPE['epochs'],eta_min=RECIPE['minimum_lr'])
        local=[]
        for epoch in range(epochs+1):
            if epoch:
                tick=time.monotonic();order=torch.randperm(len(fit));total=0.
                for start in range(0,len(order),RECIPE['batch_size']):
                    b=order[start:start+RECIPE['batch_size']]
                    loss=F.cross_entropy(net(uid[b],ids[b],offset[b]),target[b])+net.penalty(uid[b],ids[b])
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
                    with torch.no_grad():
                        for a,c in zip(ema.parameters(),net.parameters()):a.mul_(RECIPE['ema']).add_(c,alpha=1-RECIPE['ema'])
                    total+=float(loss.detach())*len(b)
                sched.step()
                if epoch%4==0:print(json.dumps(dict(stage='conditional_fit',day=day,prior=prior,epoch=epoch,loss=total/len(fit),seconds=time.monotonic()-tick)),flush=True)
            if epoch in outputs:
                new=ema.representation();np.savez(dest/('geometry_e'+str(epoch)+'.npz'),**new)
                delta=coef[0]*(cosine(new,query,vi)-initial_cosine)
                if final:
                    np.savez_compressed(dest/'evaluation.npz',correction=delta)
                    continue
                np.savez_compressed(dest/('validation_e'+str(epoch)+'.npz'),event_indices=vi,labels=labels,correction=delta)
                masses=[locked['mass']] if locked else RECIPE['masses']
                for mass in masses:
                    p=preference_probabilities(base,delta,mass)
                    row=dict(prior=prior,epoch=epoch,mass=mass,**metrics(labels,p,base,iw));local.append(row)
                print(json.dumps(dict(stage='conditional_probe',day=day,prior=prior,epoch=epoch,results=local[-len(masses):])),flush=True)
                write_json(dest/'progress.json',dict(records=local))
        write_json(dest/'result.json',dict(records=local,calibration=cal,fit_events=len(fit)))
        records+=local
    write_json(folder/'result.json',dict(records=records,calibration=cal,fit_events=len(fit),seconds=time.monotonic()-started,
               baseline=None if final else metrics(labels,base,base,iw),recipe=RECIPE,locked_selection=locked,
               max_fit_timestamp=int(raw['timestamps'][fit].max())))
    assert guard()==before


def select():
    pointer=json.loads((OUT/'incumbent_pointer.json').read_text());assert pointer==json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    results={d:json.loads((OUT/('2025-01-'+d)/'result.json').read_text()) for d in ['26','25']}
    eligible=[];rows=[]
    for a,b in zip(results['26']['records'],results['25']['records']):
        assert all(a[k]==b[k] for k in ['prior','epoch','mass'])
        delta={d:r['auc']-results[d]['baseline']['auc'] for d,r in [('26',a),('25',b)]}
        weighted_delta=np.mean([a['weighted_auc']-results['26']['baseline']['weighted_auc'],b['weighted_auc']-results['25']['baseline']['weighted_auc']])
        row=dict(prior=a['prior'],epoch=a['epoch'],mass=a['mass'],auc_26=a['auc'],auc_25=b['auc'],deltas=delta,weighted_delta=float(weighted_delta),
                 mean_near_tie_top1=(a['near_tie_top1']+b['near_tie_top1'])/2)
        rows.append(row)
        if min(delta.values())>1e-6 and weighted_delta>1e-6:eligible.append(row)
    selected=max(eligible,key=lambda x:x['weighted_delta']) if eligible else None
    write_json(OUT/'selection.json',dict(selected=selected,eligible=len(eligible),records=rows,target_auc=.838))
    write_json(OUT/'state.json',dict(stage='qualified_needs_third_origin' if selected else 'completed_no_promotion',selected=selected,updated=time.time()))
    print(json.dumps(dict(stage='conditional_selection',selected=selected,eligible=len(eligible))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--day');p.add_argument('--select',action='store_true');a=p.parse_args()
    if a.select:select()
    else:run(a.day)
