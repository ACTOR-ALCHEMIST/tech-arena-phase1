"""Third independent initialization for past-only, aligned sphere consensus."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy
import fcntl
import json
import time
import numpy as np
import torch
from scipy import sparse
from arena_ood import ROOT,write_json,sha,guard
from arena_spherical_probe import Spherical

TAG='spherical_third_v1_s2027_d8_e128'
RECIPE=dict(version=1,dimension=8,initial_seed=2027,initial_epochs=450,epochs=128,outputs=[64,128],
            seed=93831,lr=.002,minimum_lr=.0002,weight_decay=.001,ema=.99,
            negatives=8,batch_size=8192,anchor=.02,schedule='cosine128')


def model(cutoff):
    from arena_core import snapshot
    from arena_sgns_features import embedding_model
    from run import timestamp
    folder=snapshot(cutoff)/TAG;folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            assert json.loads((folder/'result.json').read_text())['recipe']==RECIPE
            return folder
        metadata=json.loads((snapshot(cutoff)/'metadata.json').read_text())
        assert metadata['max_fit_timestamp']<timestamp(cutoff)
        print(json.dumps(dict(stage='third_seed_initialization',cutoff=cutoff,recipe=RECIPE)),flush=True)
        initial=embedding_model(cutoff,8,recipe=dict(epochs=450,lr=.01,lr_decay=False,seed=2027))
        x=sparse.load_npz(snapshot(cutoff)/'interactions.npz');known_u=np.diff(x.indptr)>0
        known_i=np.asarray(x.sum(0)).ravel()>0;known_u[0]=known_i[0]=False
        pu,pi=x.nonzero();pu,pi=torch.tensor(pu,dtype=torch.long),torch.tensor(pi,dtype=torch.long)
        torch.manual_seed(RECIPE['seed']);torch.set_num_threads(2)
        net=Spherical(initial,known_u,known_i);ema=copy.deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=128,eta_min=RECIPE['minimum_lr'])
        records=[];started=time.monotonic();first_epoch=1
        if (folder/'resume.pt').exists():
            ck=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False);assert ck['recipe']==RECIPE
            net.load_state_dict(ck['net']);ema.load_state_dict(ck['ema']);opt.load_state_dict(ck['optimizer'])
            schedule.load_state_dict(ck['scheduler']);torch.random.set_rng_state(ck['rng'])
            records=ck['records'];first_epoch=ck['epoch']+1
        for epoch in range(first_epoch,129):
            tick=time.monotonic();order=torch.randperm(len(pu));total=0.
            for start in range(0,len(order),RECIPE['batch_size']):
                b=order[start:start+RECIPE['batch_size']]
                negative=torch.randint(1,len(initial['items']),(len(b),RECIPE['negatives']))
                loss=net.loss(pu[b],pi[b],negative)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
                with torch.no_grad():
                    for a,c in zip(ema.parameters(),net.parameters()):a.mul_(RECIPE['ema']).add_(c,alpha=1-RECIPE['ema'])
                total+=float(loss.detach())*len(b)
            schedule.step();records.append(dict(epoch=epoch,loss=total/len(pu),seconds=time.monotonic()-tick))
            if epoch in RECIPE['outputs']:np.savez(folder/('factors_e'+str(epoch)+'.npz'),**ema.representation())
            if epoch%8==0:
                torch.save(dict(recipe=RECIPE,epoch=epoch,net=net.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
                                scheduler=schedule.state_dict(),rng=torch.random.get_rng_state(),records=records),folder/'resume.tmp.pt')
                os.replace(folder/'resume.tmp.pt',folder/'resume.pt');write_json(folder/'training.json',records)
                print(json.dumps(dict(stage='third_sphere',cutoff=cutoff,**records[-1])),flush=True)
        write_json(folder/'result.json',dict(recipe=RECIPE,cutoff=cutoff,max_fit_timestamp=metadata['max_fit_timestamp'],
                   snapshot_metadata=metadata,positive_pairs=int(x.nnz),seconds=time.monotonic()-started,
                   model_sha256={str(e):sha(folder/('factors_e'+str(e)+'.npz')) for e in RECIPE['outputs']},
                   policy='Fixed450 SGNS and fixed128 spherical epochs, independent seed2027; all fitting clicks before cutoff; query labels never accessed.'))
    return folder


def geometry(cutoff,third_mass):
    from arena_latent_delta_export import component
    from arena_fused_sphere import fuse
    from arena_core import snapshot
    old=component(cutoff,'unitfused25_delta')[1]
    new=dict(np.load(model(cutoff)/'factors_e128.npz'));pop=np.load(snapshot(cutoff)/'stats.npz')['pop']
    combined=fuse(old,new,np.linalg.norm(old['users'],axis=1)>0,pop,1-third_mass)
    return old,combined


def campaign():
    from ctr_data import load_cache
    from arena_metric_geometry import cosine,preference_probabilities
    from arena_conditional_sphere import metrics
    from arena_validation import importance_weights
    out=ROOT/'competition/third_sphere';out.mkdir(parents=True,exist_ok=True);before=guard()
    pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    if (out/'incumbent_pointer.json').exists():assert json.loads((out/'incumbent_pointer.json').read_text())==pointer
    else:write_json(out/'incumbent_pointer.json',pointer)
    protocol=dict(recipe=RECIPE,third_masses=[.25,.5],correction_masses=[.25,.5,1.],target_auc=.838)
    write_json(out/'protocol.json',protocol);_,raw=load_cache(ROOT/'data/cache_v1','train');results={}
    for day in ['26','25']:
        frame='2025-01-'+day;write_json(out/'state.json',dict(stage='training_representation',day=day,updated=time.time()))
        model(frame);d=np.load((ROOT/pointer['csv']).parent/('validation_'+day+'.npz'))
        iw=importance_weights(frame,d['event_indices'])[0];baseline=metrics(d['labels'],d['predictions'],d['predictions'],iw);rows=[]
        for fraction in protocol['third_masses']:
            old,new=geometry(frame,fraction);delta=cosine(new,raw,d['event_indices'])-cosine(old,raw,d['event_indices'])
            np.savez_compressed(out/('correction_'+day+'_third'+str(fraction)+'.npz'),event_indices=d['event_indices'],labels=d['labels'],correction=delta)
            for mass in protocol['correction_masses']:
                p=preference_probabilities(d['predictions'],delta,mass)
                row=dict(third_mass=fraction,mass=mass,**metrics(d['labels'],p,d['predictions'],iw));rows.append(row)
                print(json.dumps(dict(stage='third_consensus_probe',day=day,**row)),flush=True)
        results[day]=dict(baseline=baseline,rows=rows);write_json(out/('result_'+day+'.json'),results[day])
    eligible=[]
    for a,b in zip(results['26']['rows'],results['25']['rows']):
        assert all(a[k]==b[k] for k in ['third_mass','mass'])
        gains=[r['auc']-results[d]['baseline']['auc'] for d,r in [('26',a),('25',b)]]
        wd=np.mean([r['weighted_auc']-results[d]['baseline']['weighted_auc'] for d,r in [('26',a),('25',b)]])
        if min(gains)>1e-6 and wd>1e-6:eligible.append(dict(third_mass=a['third_mass'],mass=a['mass'],auc_26=a['auc'],auc_25=b['auc'],weighted_delta=float(wd)))
    selected=max(eligible,key=lambda r:r['weighted_delta']) if eligible else None
    write_json(out/'selection.json',dict(selected=selected,eligible=len(eligible),target_auc=.838))
    write_json(out/'state.json',dict(stage='qualified_needs_third_origin' if selected else 'completed_no_promotion',selected=selected,updated=time.time()))
    assert guard()==before;print(json.dumps(dict(stage='third_consensus_completed',selected=selected)),flush=True)


if __name__=='__main__':campaign()
