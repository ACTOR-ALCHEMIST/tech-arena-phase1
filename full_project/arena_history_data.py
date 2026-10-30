"""Stronger time-OOF fine scores and basis-invariant target/history tokens."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy
import fcntl
import json
import time
import numpy as np
import torch
from arena_ood import ROOT,write_json,sha
from arena_two_stage_data import date_shift,raw_inputs
from arena_push_model import build_model,normalizer
from arena_tabular import predict
from arena_daily import StackRows,EventSubset

OUT=ROOT/'competition/history_rerank'
FIRST='v24_long_spectral_wide'
EPOCHS=7
TOKEN_WIDTH=31


def features(raw,ix,frame,fc):
    from arena_features import complete_features
    from arena_ood_features import extension_features
    a,an=complete_features(raw_inputs(raw),ix,frame,fc)
    b,bn=extension_features(raw_inputs(raw),ix,frame,fc,'long_spectral')
    return np.concatenate([a,b],1),an+bn


def recipe(day):
    r=json.loads((ROOT/'competition/ood/runs'/FIRST/'2025-01-26/result.json').read_text())
    cfg=copy.deepcopy(r['config']);fc=copy.deepcopy(r['feature_config'])
    fc['train_start']=max('2025-01-21',date_shift(day,-2))
    return cfg,fc


def first_path(day):
    old=ROOT/'competition/ood/runs'/FIRST/day
    return old if (old/'model.pt').exists() else OUT/'stage1'/day


def fit_first(day):
    from ctr_data import load_cache
    from run import timestamp
    path=first_path(day)
    if (path/'result.json').exists():return path
    path.mkdir(parents=True,exist_ok=True)
    cfg,fc=recipe(day);_,raw=load_cache(ROOT/'data/cache_v1','train')
    ix=np.flatnonzero((raw['timestamps']>=timestamp(fc['train_start']))&(raw['timestamps']<timestamp(day)))
    if len(ix)>180000:ix=np.sort(np.random.default_rng(42).choice(ix,180000,replace=False))
    assert len(ix) and int(raw['timestamps'][ix].max())<timestamp(day)
    np.save(path/'training_indices.npy',ix);cache=path/'working_features.npy'
    if not cache.exists():
        x,names=features(raw,ix,fc['train_start'],fc);np.save(cache,x);del x
        write_json(path/'columns.json',names)
    x=np.load(cache,mmap_mode='r');mean,scale=normalizer(x,cfg)
    torch.manual_seed(cfg['seed']);torch.set_num_threads(2)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net=build_model(x.shape[1],mean,scale,cfg).to(device);ema=copy.deepcopy(net).eval()
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=.0001)
    rng=np.random.default_rng(cfg['seed']);y=raw['labels'][ix];target=np.where(y.sum(1)>0,y.argmax(1),10)
    records=[]
    for epoch in range(1,EPOCHS+1):
        tick=time.monotonic();net.train();order=rng.permutation(len(ix));total=0.
        for start in range(0,len(ix),1024):
            take=order[start:start+1024];rows=(take[:,None]*10+np.arange(10)).ravel()
            bx=torch.tensor(np.array(x[rows]).reshape(-1,10,x.shape[1]),device=device)
            loss=torch.nn.functional.cross_entropy(net(bx),torch.tensor(target[take],device=device))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            total+=loss.item()*len(take)
            with torch.no_grad():
                for a,b in zip(ema.parameters(),net.parameters()):a.mul_(.995).add_(b,alpha=.005)
        sched.step();records.append(dict(epoch=epoch,loss=total/len(ix),seconds=time.monotonic()-tick))
        print(json.dumps(dict(stage='history_fine_fit',day=day,**records[-1])),flush=True)
        write_json(path/'training.json',records)
    torch.save(dict(state_dict=ema.cpu().state_dict(),width=x.shape[1],config=cfg),path/'model.pt')
    write_json(path/'result.json',dict(config=cfg,feature_config=fc,best_epoch=EPOCHS,training_events=len(ix),
               max_fit_timestamp=int(raw['timestamps'][ix].max()),policy='Fixed7 epochs, strictly before day; no query labels or metrics accessed.'))
    del x;write_json(path/'cache_cleanup.json',dict(path=str(cache.relative_to(ROOT)),sha256=sha(cache),bytes=cache.stat().st_size))
    cache.unlink();return path


def load_first(path,device):
    c=torch.load(path/'model.pt',map_location='cpu',weights_only=False);s=c['state_dict']
    net=build_model(c['width'],s['mean'].numpy(),s['scale'].numpy(),c['config'])
    net.load_state_dict(s);return net.to(device).eval()


def prepare(day):
    from ctr_data import load_cache,temporal_split
    from run import timestamp
    folder=OUT/'inputs'/day;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if all((folder/f).exists() for f in ['protocol.json','scores.npz','features.npy']):return folder
        _,raw=load_cache(ROOT/'data/cache_v1','train');_,ix=temporal_split(raw,timestamp(day))
        path=fit_first(day);_,fc=recipe(day)
        if not (folder/'features.npy').exists():
            x,names=features(raw,ix,day,fc);np.save(folder/'features.npy',x);del x
            write_json(folder/'columns.json',names)
        x=np.load(folder/'features.npy',mmap_mode='r')
        torch.set_num_threads(2);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        p=predict(load_first(path,device),x,device)
        if (path/'validation.npz').exists():
            old=np.load(path/'validation.npz');np.testing.assert_array_equal(old['event_indices'],ix)
            assert np.max(np.abs(p-old['predictions']))<2e-5
        r=json.loads((path/'result.json').read_text())
        fitting=np.load(path/'training_indices.npy') if (path/'training_indices.npy').exists() else np.load(ROOT/r['feature_dataset']/'labels.npz')['train_indices']
        assert int(raw['timestamps'][fitting].max())<int(raw['timestamps'][ix].min())
        np.savez_compressed(folder/'scores.npz',event_indices=ix,predictions=p,labels=raw['labels'][ix])
        write_json(folder/'protocol.json',dict(day=day,width=244,events=len(ix),ranker=str(path.relative_to(ROOT)),
                   ranker_sha256=sha(path/'model.pt'),max_fit_timestamp=int(raw['timestamps'][fitting].max()),
                   min_query_timestamp=int(raw['timestamps'][ix].min()),policy='First query per user; all fitting targets and representation data before origin.'))
        print(json.dumps(dict(stage='history_inputs_ready',day=day,events=len(ix))),flush=True)
    return folder


class GeometryBank:
    def __init__(self,raw,overrides=None):
        self.raw=raw_inputs(raw);self.overrides=overrides or {};self.cache={}

    def get(self,frame):
        if frame not in self.cache:
            if frame in self.overrides:reps=self.overrides[frame]
            else:
                from arena_spherical_repr import model as old
                from arena_spherical_long import model as long
                from arena_latent_delta_export import component
                reps=[dict(np.load(old(frame)/'factors_e32.npz')),dict(np.load(long(frame)/'factors_e128.npz')),component(frame,'unitfused25_delta')[1]]
            values=[]
            for rep in reps:
                u,v=np.asarray(rep['users'],dtype='f'),np.asarray(rep['items'],dtype='f')
                un,vn=np.linalg.norm(u,axis=1),np.linalg.norm(v,axis=1)
                values.append((u/un[:,None].clip(1e-8),v/vn[:,None].clip(1e-8),un,vn))
            self.cache[frame]=values
        return self.cache[frame]

    def render(self,frame,ix,prob):
        from run import timestamp
        assert np.all(self.raw['timestamps'][ix]>=timestamp(frame))
        uid=self.raw['users'][ix];iid=self.raw['items'][ix];hist=self.raw['history'][ix]
        age=np.asarray(self.raw['age_hours'][ix]);visible=hist!=0;length=visible.sum(1)
        q=prob/prob.sum(1,keepdims=True).clip(1e-8)
        sims=[];relative=[];uhs=[];cohs=[];cns=[];hns=[];uns=[];pair=None
        for u,v,un,vn in self.get(frame):
            h=v[hist];c=v[iid];uu=u[uid]
            sim=np.einsum('bid,btd->bit',c,h)
            rel=sim-np.einsum('bi,bit->bt',q,sim)[:,None,:]
            uh=np.einsum('bd,btd->bt',uu,h)
            pooled=(h*visible[:,:,None]).sum(1)
            coherence=(np.einsum('btd,bd->bt',h,pooled)-(h*h).sum(2))/(length-1).clip(1)[:,None]
            sims.append(sim);relative.append(rel)
            expand=lambda a:np.broadcast_to(a[:,None,:],sim.shape)
            uhs.append(expand(uh));cohs.append(expand(coherence))
            cns.append(np.broadcast_to(vn[iid][:,:,None],sim.shape))
            hns.append(expand(vn[hist]));uns.append(np.broadcast_to(un[uid,None,None],sim.shape))
            pair=np.einsum('bid,bjd->bij',c,c)
        minutes=self.raw['timestamps'][ix]//60
        hm=minutes[:,None]-np.rint(age*60).astype(np.int64)
        same_bucket=(hm%1440//360)==(minutes%1440//360)[:,None]
        same_weekend=(((hm//1440+3)%7)>=5)==(((minutes//1440+3)%7)>=5)[:,None]
        expand=lambda a:np.broadcast_to(a[:,None,:],(len(ix),10,20))
        times=[expand(np.log1p(age)/np.log1p(720))]+[expand(np.exp(-age/t)) for t in [6,24,72,168]]
        times += [expand(np.sin(age*2*np.pi/24)),expand(np.cos(age*2*np.pi/24)),
                  (iid[:,:,None]==hist[:,None,:]).astype('f'),expand(same_bucket),expand(same_weekend)]
        tokens=np.stack(sims+relative+uhs+cohs+cns+hns+uns+times,axis=3).astype('f')
        assert tokens.shape==(len(ix),10,20,TOKEN_WIDTH) and np.isfinite(tokens).all()
        return tokens,visible,pair.astype('f')


class Data:
    def __init__(self,cutoff,raw,training=False):
        from run import timestamp
        frames=[date_shift(cutoff,-2),date_shift(cutoff,-1)] if training else [cutoff]
        self.bank=GeometryBank(raw);parts=[];indices=[];labels=[];scores=[];origins=[];self.folders=[]
        for frame in frames:
            folder=prepare(frame);self.folders.append(folder);s=np.load(folder/'scores.npz');ix=s['event_indices']
            pos=np.flatnonzero(raw['timestamps'][ix]<timestamp(cutoff)) if training else np.arange(len(ix))
            ix=ix[pos]
            if training:
                assert int(raw['timestamps'][ix].max())<timestamp(cutoff)
                hm=raw['timestamps'][ix,None]//60-np.rint(raw['age_hours'][ix]*60).astype(np.int64)
                assert np.all(hm[raw['history'][ix]!=0]<timestamp(frame)//60)
            parts.append(EventSubset(np.load(folder/'features.npy',mmap_mode='r'),pos))
            indices.append(ix);labels.append(s['labels'][pos]);scores.append(s['predictions'][pos]);origins.extend([frame]*len(ix))
        self.features=StackRows(*parts);self.indices=np.concatenate(indices);self.labels=np.concatenate(labels)
        self.prob=np.concatenate(scores);self.frames=np.asarray(origins)

    def batch(self,take,device):
        rows=(take[:,None]*10+np.arange(10)).ravel()
        x=self.features[rows].reshape(-1,10,self.features.shape[1]);p=self.prob[take]
        t=np.empty((len(take),10,20,TOKEN_WIDTH),'f');m=np.empty((len(take),20),bool);pair=np.empty((len(take),10,10),'f')
        for frame in np.unique(self.frames[take]):
            mask=self.frames[take]==frame;t[mask],m[mask],pair[mask]=self.bank.render(str(frame),self.indices[take[mask]],p[mask])
        return tuple(torch.tensor(a,device=device) for a in [x,p,t,m,pair])
