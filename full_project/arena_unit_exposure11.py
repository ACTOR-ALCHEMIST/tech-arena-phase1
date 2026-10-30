"""Full-prefix 11-class exposure likelihood with fixed calibrated head.

Unlike the clicked-slate conditional update, no-click events provide vector
gradients too. Unit norms and support-weighted priors constrain the geometry;
there are no freely learned per-ID propensity intercepts. Inference compares
directions to the accepted joint geometry and keeps parent event click mass.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy
import json
import time
import numpy as np
import torch
from scipy import sparse
from scipy.special import logit
from torch import nn
from torch.nn import functional as F
from arena_ood import ROOT,guard,write_json,sha
from arena_conditional_sphere import ConditionalSphere,covariates,unit,metrics
from arena_metric_geometry import cosine,preference_probabilities

OUT=ROOT/'competition/unit_exposure11'
RECIPE=dict(version=1,epochs=16,outputs=[4,8,16],priors=[1.,10.],masses=[.25,.5,1.],
            seed=84001,lr=.001,minimum_lr=.0001,batch_size=4096,ema=.98,
            head_max_events=100000,head_iterations=60,head_ridge=.003,
            reference='conditional_full/prior1.0/geometry_e8.npz',reference_mass=.25,
            objective='all_past_events_11_class_real_exposures')


def targets(labels):
    y=np.asarray(labels);assert np.all(y.sum(1)<=1)
    return np.where(y.sum(1)>0,y.argmax(1),10).astype('int64')


def global_features(raw,ix,stats):
    return np.concatenate([stats['ub'][raw['users'][ix],None],raw['context'][ix]],1).astype('f')


def globals_design(x,head):
    return np.clip((x-np.asarray(head['mean']))/np.asarray(head['scale']),-6,6)*np.asarray(head['known'])


def fit_head(day,raw,fit):
    from arena_two_stage_data import date_shift
    from arena_latent_delta_export import component
    from arena_core import snapshot
    from run import timestamp
    path=OUT/day/'head.json'
    if path.exists():return json.loads(path.read_text())
    prefix=date_shift(day,-2);ix=fit[raw['timestamps'][fit]>=timestamp(prefix)]
    if len(ix)>RECIPE['head_max_events']:ix=np.sort(np.random.default_rng(RECIPE['seed']).choice(ix,RECIPE['head_max_events'],replace=False))
    assert len(ix) and int(raw['timestamps'][ix].min())>=timestamp(prefix) and int(raw['timestamps'][ix].max())<timestamp(day)
    rep=component(prefix,'unitfused25_delta')[1];stats=dict(np.load(snapshot(prefix)/'stats.npz'))
    x=covariates(raw,ix,rep,stats['ib']);glob=global_features(raw,ix,stats)
    mean=glob.mean(0);sd=glob.std(0);scale=sd.clip(1e-5);known=(sd>1e-5).astype('f')
    desc=dict(mean=mean.tolist(),scale=scale.tolist(),known=known.tolist())
    g=globals_design(glob,desc);y=targets(raw['labels'][ix])
    old=json.loads((ROOT/'competition/conditional_full'/day/'calibration.json').read_text())
    prior=np.zeros(15,dtype='f');prior[:7]=old['coefficients'];prior[7]=2*sd[0]
    z=(x*prior[:7]).sum(2)+(g@prior[7:14])[:,None]
    lse=np.logaddexp.reduce(z,axis=1);prior[14]=lse.mean()-logit((y!=10).mean())
    initial=prior.copy();initial[0]=logit(np.clip((prior[0]-1)/11,1e-5,1-1e-5))
    beta=nn.Parameter(torch.tensor(initial));x=torch.tensor(x);g=torch.tensor(g,dtype=torch.float32);y=torch.tensor(y)
    prior_tensor=torch.tensor(prior)
    opt=torch.optim.LBFGS([beta],lr=.5,max_iter=RECIPE['head_iterations'],line_search_fn='strong_wolfe')
    def coefficients():return torch.cat([1+11*beta[:1].sigmoid(),beta[1:]])
    def closure():
        opt.zero_grad();c=coefficients();inside=(x*c[:7]).sum(2)+(g@c[7:14])[:,None]
        logits=torch.cat([inside,c[14].expand(len(x),1)],1)
        loss=F.cross_entropy(logits,y)+RECIPE['head_ridge']*(c-prior_tensor).square().mean()
        loss.backward();return loss
    opt.step(closure);c=coefficients().detach().numpy()
    result=dict(**desc,coefficients=c.tolist(),prefix=prefix,events=len(ix),click_events=int((y.numpy()!=10).sum()),
                first_fit_timestamp=int(raw['timestamps'][ix].min()),max_fit_timestamp=int(raw['timestamps'][ix].max()),
                policy='All head targets after an earlier geometry anchor and strictly before cutoff. Constant global features disabled; bounded temperature1..12.')
    write_json(path,result);return result


def offset_features(raw,ix,rep,stats,head):
    c=np.asarray(head['coefficients'],dtype='f');x=covariates(raw,ix,rep,stats['ib'])
    global_offset=globals_design(global_features(raw,ix,stats),head)@c[7:14]
    return ((x[:,:,1:]*c[1:7]).sum(2)+global_offset[:,None]).astype('f')


def exposure_logits(net,uid,ids,offset,outside):
    score=net(uid,ids,offset)
    return torch.cat([score,torch.full((len(uid),1),float(outside),dtype=score.dtype,device=score.device)],1)


def fit(day,prior,raw):
    from run import timestamp
    from arena_core import snapshot
    from arena_latent_delta_export import component
    folder=OUT/day/('prior'+str(prior));folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return folder
    ix=np.flatnonzero(raw['timestamps']<timestamp(day));assert len(ix) and int(raw['timestamps'][ix].max())<timestamp(day)
    head=fit_head(day,raw,ix);rep=component(day,'unitfused25_delta')[1];rep={k:unit(rep[k]) for k in ['users','items']}
    stats=dict(np.load(snapshot(day)/'stats.npz'));offset=torch.tensor(offset_features(raw,ix,rep,stats,head))
    uid=torch.tensor(np.array(raw['users'][ix]),dtype=torch.long);ids=torch.tensor(np.array(raw['items'][ix]),dtype=torch.long)
    y=torch.tensor(targets(raw['labels'][ix]));counts=(np.bincount(raw['users'][ix],minlength=len(rep['users'])),np.bincount(raw['items'][ix].ravel(),minlength=len(rep['items'])))
    graph=sparse.load_npz(snapshot(day)/'interactions.npz')
    support=(np.diff(graph.indptr).clip(1).astype('f'),np.asarray(graph.sum(0)).ravel().clip(1).astype('f'))
    torch.manual_seed(RECIPE['seed']);net=ConditionalSphere(rep,counts,support,head['coefficients'][0],prior);ema=copy.deepcopy(net).eval()
    opt=torch.optim.AdamW(net.parameters(),lr=RECIPE['lr'],weight_decay=0)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=RECIPE['epochs'],eta_min=RECIPE['minimum_lr']);records=[];started=time.monotonic()
    for epoch in range(1,RECIPE['epochs']+1):
        tick=time.monotonic();order=torch.randperm(len(ix));total=0.
        for start in range(0,len(order),RECIPE['batch_size']):
            b=order[start:start+RECIPE['batch_size']]
            loss=F.cross_entropy(exposure_logits(net,uid[b],ids[b],offset[b],head['coefficients'][14]),y[b])+net.penalty(uid[b],ids[b])
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            with torch.no_grad():
                for a,c in zip(ema.parameters(),net.parameters()):a.mul_(RECIPE['ema']).add_(c,alpha=1-RECIPE['ema'])
            total+=float(loss.detach())*len(b)
        schedule.step();records.append(dict(epoch=epoch,loss=total/len(ix),seconds=time.monotonic()-tick));write_json(folder/'training.json',records)
        if epoch in RECIPE['outputs']:np.savez(folder/('geometry_e'+str(epoch)+'.npz'),**ema.representation())
        if epoch%4==0:print(json.dumps(dict(stage='exposure11_fit',day=day,prior=prior,**records[-1])),flush=True)
    write_json(folder/'result.json',dict(recipe=RECIPE,prior=prior,training_events=len(ix),clicked_events=int((y.numpy()!=10).sum()),
               no_click_events=int((y.numpy()==10).sum()),max_fit_timestamp=int(raw['timestamps'][ix].max()),seconds=time.monotonic()-started,
               head_sha256=sha(OUT/day/'head.json'),policy='All past actual events, including no-clicks; fixed calibrated15-parameter score head, unit user/item vectors and once-per-ID support priors.'))
    return folder


def campaign():
    from ctr_data import load_cache
    from arena_validation import importance_weights
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);before=guard()
    pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text());write_json(OUT/'incumbent_pointer.json',pointer);write_json(OUT/'recipe.json',RECIPE)
    assert not (OUT/'selection.json').exists(),'Completed probe is immutable.'
    _,raw=load_cache(ROOT/'data/cache_v1','train');results={}
    for day in ['26','25']:
        write_json(OUT/'state.json',dict(stage='training',day=day,updated=time.time()))
        frame='2025-01-'+day;parent=np.load((ROOT/pointer['csv']).parent/('validation_'+day+'.npz'));ix=parent['event_indices']
        old=dict(np.load(ROOT/'competition/conditional_full'/frame/'prior1.0/geometry_e8.npz'))
        scale=RECIPE['reference_mass']*json.loads((ROOT/'competition/conditional_full'/frame/'calibration.json').read_text())['coefficients'][0]
        old_cos=cosine(old,raw,ix);iw=importance_weights(frame,ix)[0];baseline=metrics(parent['labels'],parent['predictions'],parent['predictions'],iw);rows=[]
        for prior in RECIPE['priors']:
            folder=fit(frame,prior,raw)
            for epoch in RECIPE['outputs']:
                new=dict(np.load(folder/('geometry_e'+str(epoch)+'.npz')));delta=scale*(cosine(new,raw,ix)-old_cos)
                np.savez_compressed(folder/('validation_e'+str(epoch)+'.npz'),event_indices=ix,labels=parent['labels'],correction=delta)
                for mass in RECIPE['masses']:
                    p=preference_probabilities(parent['predictions'],delta,mass)
                    row=dict(prior=prior,epoch=epoch,mass=mass,**metrics(parent['labels'],p,parent['predictions'],iw));rows.append(row)
                    print(json.dumps(dict(stage='exposure11_probe',day=day,**row)),flush=True)
        results[day]=dict(baseline=baseline,rows=rows);write_json(OUT/('result_'+day+'.json'),results[day])
        if day=='26' and max(r['auc'] for r in rows)<=baseline['auc']+1e-6:break
    eligible=[]
    if '25' in results:
        for a,b in zip(results['26']['rows'],results['25']['rows']):
            assert all(a[k]==b[k] for k in ['prior','epoch','mass'])
            gain=[r['auc']-results[d]['baseline']['auc'] for d,r in [('26',a),('25',b)]]
            wd=np.mean([r['weighted_auc']-results[d]['baseline']['weighted_auc'] for d,r in [('26',a),('25',b)]])
            if min(gain)>1e-6 and wd>1e-6:eligible.append(dict(prior=a['prior'],epoch=a['epoch'],mass=a['mass'],auc_26=a['auc'],auc_25=b['auc'],weighted_delta=float(wd)))
    selected=max(eligible,key=lambda r:r['weighted_delta']) if eligible else None
    write_json(OUT/'selection.json',dict(selected=selected,eligible=len(eligible),primary_only='25' not in results,target_auc=.838))
    write_json(OUT/'state.json',dict(stage='qualified_needs_third_origin' if selected else 'completed_no_promotion',selected=selected,updated=time.time()))
    lines=['# 加入整轮无点击曝光的单位球偏好训练','',
           '先用更早表示校准15个全局参数和无点击选项，再固定头部，使用全部历史真实事件训练单位用户/物品方向。无点击事件也产生表示梯度。',
           '向原模型只转移方向差，不直接替换事件点击概率；与已经采用的联合几何更新做差。常数时间特征禁用，三切分门槛沿用。','',
           '| 先验 | 轮数 | 幅度 | Jan26 | Jan25 |','|---|---|---|---:|---:|']
    for i,a in enumerate(results['26']['rows']):
        second=f"{results['25']['rows'][i]['auc']:.9f}" if '25' in results else '主切分未通过，未训练'
        lines.append(f'| {a["prior"]} | {a["epoch"]} | {a["mass"]} | {a["auc"]:.9f} | {second} |')
    lines+=['',f'合格 {len(eligible)} 组；未合格则保留V31。全局经验先验复用了拟合期统计，不视为校准后的贝叶斯置信度。目标0.838尚未达到。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));assert guard()==before;print(json.dumps(dict(stage='exposure11_completed',selected=selected)),flush=True)


if __name__=='__main__':campaign()
