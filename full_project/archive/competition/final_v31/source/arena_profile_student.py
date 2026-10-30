"""Distill a source-only user preference teacher into a history Transformer.

The student receives no user ID or candidate labels. It learns from source
histories to reconstruct the stronger ID teacher's preference direction, then
contributes only evidence beyond an unweighted history-vector mean.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_ood import ROOT,guard,write_json
from arena_metric_geometry import preference_probabilities

OUT=ROOT/'competition/profile_student'
RECIPE=dict(version=1,width=32,heads=4,layers=1,epochs=16,outputs=[8,16],max_events=120000,
            seed=83931,batch_size=512,lr=.001,minimum_lr=.0001,weight_decay=.03,
            dropout=.1,history_dropout=.2,ema=.99,masses=[.1,.25,.5],distillation='unit_user_direction')


class ProfileStudent(nn.Module):
    def __init__(self,items,config=None):
        super().__init__();self.config=config or RECIPE;w=self.config['width'];d=items.shape[1]
        v=torch.tensor(np.asarray(items),dtype=torch.float32);v=F.normalize(v,dim=1,eps=1e-6);v[0]=0
        self.register_buffer('items',v)
        self.register_buffer('known_items',torch.linalg.vector_norm(v,dim=1)>0)
        self.input=nn.Linear(d+4,w);self.position=nn.Embedding(21,w,padding_idx=0)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(w,self.config['heads'],w*2,self.config['dropout'],
                                  batch_first=True,norm_first=True) for _ in range(self.config['layers'])])
        self.output=nn.Sequential(nn.Linear(w*3+2,64),nn.SiLU(),nn.Dropout(self.config['dropout']),nn.Linear(64,d))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,history,age,augment=False):
        visible=(history!=0)&self.known_items[history]
        if augment:
            original=visible
            visible=visible&(torch.rand(visible.shape,device=history.device)>=self.config['history_dropout'])
            restore=original.any(1)&~visible.any(1)
            last=(original*torch.arange(1,history.shape[1]+1,device=history.device)).argmax(1)
            visible[torch.arange(len(history),device=history.device)[restore],last[restore]]=True
        h=self.items[history]*visible[...,None];count=visible.sum(1);has=count>0
        base=h.sum(1)/count.clamp_min(1)[:,None]
        rank=visible.cumsum(1);pos=torch.where(visible,(count[:,None]-rank+1).clamp(0,20),0)
        t=age.clamp(0,24*365)
        times=torch.stack([t.log1p()/np.log1p(720),torch.exp(-t/72),torch.sin(t*2*np.pi/24),torch.cos(t*2*np.pi/24)],2)
        z=(self.input(torch.cat([h,times],2))+self.position(pos))*visible[...,None]
        safe=visible.clone();safe[:,0]|=~has
        for layer in self.layers:z=layer(z,src_key_padding_mask=~safe)
        mean=(z*visible[...,None]).sum(1)/count.clamp_min(1)[:,None]
        peak=z.masked_fill(~visible[...,None],-1e4).max(1).values;peak=torch.where(has[:,None],peak,0)
        last=(z*(visible&(rank==count[:,None]))[...,None]).sum(1)
        support=torch.stack([count.float()/20,torch.linalg.vector_norm(base,dim=1)],1)
        delta=self.output(torch.cat([mean,peak,last,support],1))
        q=F.normalize(base+.5*torch.tanh(delta),dim=1,eps=.1)*has[:,None]
        return q,F.normalize(base,dim=1,eps=.1)*has[:,None]


@torch.inference_mode()
def correction(net,raw,ix):
    net.eval();parts=[]
    for start in range(0,len(ix),512):
        b=ix[start:start+512]
        history=torch.tensor(np.array(raw['history'][b]),dtype=torch.long);age=torch.tensor(np.array(raw['age_hours'][b]))
        q,base=net(history,age);v=net.items[torch.tensor(np.array(raw['items'][b]),dtype=torch.long)]
        parts.append(((q-base)[:,None]*v).sum(2).numpy())
    return np.concatenate(parts)


def fit(day,raw):
    from arena_latent_delta_export import component
    from run import timestamp
    path=OUT/day;path.mkdir(parents=True,exist_ok=True)
    if (path/'result.json').exists():return path
    rep=component(day,'unitfused25_delta')[1]
    users=np.asarray(rep['users'],dtype='f');users/=np.linalg.norm(users,axis=1,keepdims=True).clip(1e-8)
    indices=np.flatnonzero((raw['timestamps']<timestamp(day))&(raw['history']!=0).any(1)&(np.linalg.norm(users[raw['users']],axis=1)>0))
    if len(indices)>RECIPE['max_events']:
        indices=np.sort(np.random.default_rng(RECIPE['seed']).choice(indices,RECIPE['max_events'],replace=False))
    assert int(raw['timestamps'][indices].max())<timestamp(day)
    torch.manual_seed(RECIPE['seed']);torch.set_num_threads(2)
    net=ProfileStudent(rep['items']);ema=copy.deepcopy(net).eval()
    opt=torch.optim.AdamW(net.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=RECIPE['epochs'],eta_min=RECIPE['minimum_lr'])
    history=torch.tensor(np.array(raw['history'][indices]),dtype=torch.long)
    ages=torch.tensor(np.array(raw['age_hours'][indices]));target=torch.tensor(users[raw['users'][indices]])
    records=[];started=time.monotonic()
    for epoch in range(1,RECIPE['epochs']+1):
        tick=time.monotonic();net.train();order=torch.randperm(len(indices));total=0.
        for start in range(0,len(indices),RECIPE['batch_size']):
            b=order[start:start+RECIPE['batch_size']]
            q,_=net(history[b],ages[b],True)
            loss=(1-(q*target[b]).sum(1)).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
            with torch.no_grad():
                for a,c in zip(ema.parameters(),net.parameters()):a.mul_(RECIPE['ema']).add_(c,alpha=1-RECIPE['ema'])
            total+=float(loss.detach())*len(b)
        schedule.step();row=dict(epoch=epoch,loss=total/len(indices),seconds=time.monotonic()-tick);records.append(row)
        write_json(path/'training.json',records)
        print(json.dumps(dict(stage='profile_distillation',day=day,**row)),flush=True)
        if epoch in RECIPE['outputs']:torch.save(dict(state_dict=ema.state_dict(),config=RECIPE),path/('e'+str(epoch)+'.pt'))
    write_json(path/'result.json',dict(recipe=RECIPE,training_events=len(indices),max_fit_timestamp=int(raw['timestamps'][indices].max()),
               parameters=sum(p.numel() for p in net.parameters()),seconds=time.monotonic()-started,
               policy='Only histories of source events before cutoff. User IDs index a source-only teacher direction for training targets; student input contains no user ID or click labels. No held-out query targets used.'))
    return path


def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);net=ProfileStudent(ck['state_dict']['items'].numpy(),ck['config'])
    net.load_state_dict(ck['state_dict']);return net.eval()


def campaign():
    from ctr_data import load_cache
    from arena_conditional_sphere import metrics
    from arena_validation import importance_weights
    OUT.mkdir(parents=True,exist_ok=True);before=guard();pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    assert not (OUT/'selection.json').exists(),'Completed experiment is immutable.'
    write_json(OUT/'incumbent_pointer.json',pointer);write_json(OUT/'recipe.json',RECIPE)
    _,raw=load_cache(ROOT/'data/cache_v1','train');results={}
    for day in ['26','25']:
        frame='2025-01-'+day;write_json(OUT/'state.json',dict(stage='training',day=day,updated=time.time()))
        path=fit(frame,raw);d=np.load((ROOT/pointer['csv']).parent/('validation_'+day+'.npz'))
        iw=importance_weights(frame,d['event_indices'])[0];baseline=metrics(d['labels'],d['predictions'],d['predictions'],iw);rows=[]
        for epoch in RECIPE['outputs']:
            delta=correction(load(path/('e'+str(epoch)+'.pt')),raw,d['event_indices'])
            np.savez_compressed(path/('correction_e'+str(epoch)+'.npz'),event_indices=d['event_indices'],labels=d['labels'],correction=delta)
            for mass in RECIPE['masses']:
                p=preference_probabilities(d['predictions'],delta,mass)
                row=dict(epoch=epoch,mass=mass,**metrics(d['labels'],p,d['predictions'],iw));rows.append(row)
                print(json.dumps(dict(stage='profile_probe',day=day,**row)),flush=True)
        results[day]=dict(baseline=baseline,rows=rows);write_json(OUT/('result_'+day+'.json'),results[day])
    eligible=[]
    for a,b in zip(results['26']['rows'],results['25']['rows']):
        assert a['epoch']==b['epoch'] and a['mass']==b['mass']
        ds=[r['auc']-results[d]['baseline']['auc'] for d,r in [('26',a),('25',b)]]
        wd=np.mean([r['weighted_auc']-results[d]['baseline']['weighted_auc'] for d,r in [('26',a),('25',b)]])
        if min(ds)>1e-6 and wd>1e-6:eligible.append(dict(epoch=a['epoch'],mass=a['mass'],auc_26=a['auc'],auc_25=b['auc'],weighted_delta=float(wd)))
    selected=max(eligible,key=lambda r:r['weighted_delta']) if eligible else None
    write_json(OUT/'selection.json',dict(selected=selected,eligible=len(eligible),target_auc=.838))
    write_json(OUT/'state.json',dict(stage='qualified_needs_third_origin' if selected else 'completed_no_promotion',selected=selected,updated=time.time()))
    assert guard()==before;print(json.dumps(dict(stage='profile_completed',selected=selected)),flush=True)


if __name__=='__main__':campaign()
