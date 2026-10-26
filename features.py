"""V31 inference features, extracted from the frozen release; no training code."""
import numpy as np
from numba import njit, prange, set_num_threads
from scipy import sparse
from threadpoolctl import threadpool_limits
from implicit.cpu.als import AlternatingLeastSquares

# The caller loads the packaged assets once, then builds the fixed 323 columns.
STORE = None
ASSET_DIR = None


def rep(name):
    return {k.split('__', 1)[1]: v for k, v in STORE.items() if k.startswith(name+'__')}


def load_snapshot(_):
    return sparse.load_npz(ASSET_DIR/'co_counts.npz'), rep('stats')


def latent_model(cutoff, factors, alpha=10, regularization=.1):
    values = rep('als'+str(factors))
    model = AlternatingLeastSquares(factors=factors, alpha=alpha, regularization=regularization,
                                   iterations=25, num_threads=6, random_state=42)
    model.user_factors, model.item_factors = values['users'], values['items']
    return model


def embedding_model(cutoff, factors=16, seed=42, epochs=None, recipe=None):
    seed = (recipe or {}).get('seed', seed)
    return rep('sgns'+str(factors)+'_s'+str(seed))


def context_stats(cutoff):
    return rep('context')


@njit(parallel=True, cache=True)
def co_features(indptr, columns, values, pop, candidates, histories, ages):
    n = len(candidates)
    out = np.zeros((n, 10, 9), dtype=np.float32)
    inv = 1/np.sqrt(np.maximum(pop, 1))
    for e in prange(n):
        for j in range(10):
            item = candidates[e, j]
            a, b = indptr[item], indptr[item+1]
            for k in range(20):
                h = histories[e, k]
                if h == 0:
                    continue
                lo, hi = a, b
                while lo < hi:
                    mid = (lo+hi)//2
                    if columns[mid] < h:
                        lo = mid+1
                    else:
                        hi = mid
                if lo >= b or columns[lo] != h:
                    continue
                count = values[lo]
                cosine = count*inv[item]*inv[h]
                age_days = ages[e, k]/24
                out[e, j, 0] += cosine
                out[e, j, 1] = max(out[e, j, 1], cosine)
                out[e, j, 2] += cosine*np.exp(-age_days/0.5)
                out[e, j, 3] += cosine*np.exp(-age_days/3)
                out[e, j, 4] += cosine*np.exp(-age_days/30)
                out[e, j, 5] += cosine*count/(count+3)
                out[e, j, 6] += np.log1p(count)
                out[e, j, 7] += count/max(pop[h], 1)
                if k == 19 or histories[e, k+1] == 0:
                    out[e, j, 8] = cosine
    return out

def base_features(data, indices, path):
    c, stats = load_snapshot(path)
    items = np.array(data["items"][indices])
    users = np.array(data["users"][indices])
    h = np.array(data["history"][indices])
    ages = np.array(data["age_hours"][indices])
    cf = co_features(c.indptr, c.indices, c.data, stats["pop"].astype(np.float32), items, h, ages)
    del c
    names = ["cf_cos_sum", "cf_cos_max", "cf_recent_12h", "cf_recent_3d", "cf_recent_30d",
             "cf_shrink3", "cf_log_count", "cf_conditional", "cf_last"]
    parts = [cf]
    def add(name, value):
        names.append(name)
        if value.ndim == 1:
            value = np.broadcast_to(value[:, None], items.shape)
        parts.append(value[..., None].astype(np.float32))
    prior = float(np.log(stats["ctr"]/(1-stats["ctr"])))+stats["ub"][users, None]+stats["ib"][items]
    add("prior_logit", prior)
    add("user_bias", stats["ub"][users]);add("item_bias", stats["ib"][items])
    for name in ("uc", "ui"):
        add("log_"+name, np.log1p(stats[name][users]))
    for name in ("ic", "ii", "pop"):
        add("log_"+name, np.log1p(stats[name][items]))
    mask = (items[:, :, None] == h[:, None, :]) & (h[:, None, :] != 0)
    count = mask.sum(2).astype(np.float32)
    add("repeat_any", (count>0).astype(np.float32));add("repeat_count", count)
    for tau in (0.5, 3, 30):
        add("repeat_decay_"+str(tau), (mask*np.exp(-ages[:, None, :]/(24*tau))).sum(2))
    for k, name in enumerate(("hour_sin", "hour_cos", "weekend", "history_length", "last_gap", "history_span")):
        add(name, np.array(data["context"][indices, k]))
    for k in (0, 1, 3, 5):
        add(names[k]+"_relative", cf[:, :, k]-cf[:, :, k].mean(1, keepdims=True))
        add(names[k]+"_rank", np.argsort(np.argsort(cf[:, :, k], axis=1), axis=1).astype(np.float32)/9)
    add("item_bias_relative", stats["ib"][items]-stats["ib"][items].mean(1, keepdims=True))
    matrix = np.concatenate(parts, axis=2).reshape(-1, len(names))
    return matrix, names, prior, cf

def latent_features(data,indices,cutoff,factors=32,alpha=10,regularization=0.1):
    model=latent_model(cutoff,factors,alpha,regularization)
    n=len(indices)
    items=np.array(data["items"][indices]);users=np.array(data["users"][indices])
    history=np.array(data["history"][indices]);ages=np.array(data["age_hours"][indices])
    keep=history.ravel()!=0
    q=sparse.coo_matrix((np.ones(keep.sum(),dtype=np.float32),
        (np.repeat(np.arange(n),20)[keep],history.ravel()[keep])),shape=(n,len(model.item_factors))).tocsr()
    q.data=1+np.log(q.data)
    with threadpool_limits(limits=1,user_api="blas"):
        folded=model.recalculate_user(np.arange(n),q)
    known=model.user_factors[users].copy()
    unknown=(known*known).sum(1)<1e-12
    known[unknown]=folded[unknown]
    names=[f"als{factors}_{s}" for s in ("known_dot","fold_dot","known_cos","fold_cos","known_norm","fold_norm",
            "item_norm","history_dot_mean","history_cos_mean","history_cos_max","history_cos_recent",
            "known_relative","fold_relative","cos_relative","fold_cos_relative")]
    out=np.zeros((n,10,len(names)),dtype=np.float32)
    for start in range(0,n,4096):
        end=min(n,start+4096)
        v=model.item_factors[items[start:end]]
        u=known[start:end];f=folded[start:end]
        un=np.linalg.norm(u,axis=1).clip(1e-6);fn=np.linalg.norm(f,axis=1).clip(1e-6)
        vn=np.linalg.norm(v,axis=2).clip(1e-6)
        ud=(v*u[:,None]).sum(2);fd=(v*f[:,None]).sum(2)
        uc=ud/(un[:,None]*vn);fc=fd/(fn[:,None]*vn)
        h=model.item_factors[history[start:end]]
        hn=np.linalg.norm(h,axis=2).clip(1e-6)
        valid=history[start:end]!=0
        hd=np.einsum("bik,bjk->bij",v,h)
        hc=hd/(vn[:,:,None]*hn[:,None,:]);hc*=valid[:,None,:]
        length=valid.sum(1).clip(1)
        recency=np.exp(-ages[start:end]/72)*valid
        recent=(hc*recency[:,None,:]).sum(2)/recency.sum(1).clip(1e-6)[:,None]
        arrays=[ud,fd,uc,fc,np.broadcast_to(un[:,None],ud.shape),np.broadcast_to(fn[:,None],ud.shape),vn,
                (hd*valid[:,None,:]).sum(2)/length[:,None],hc.sum(2)/length[:,None],hc.max(2),recent,
                ud-ud.mean(1,keepdims=True),fd-fd.mean(1,keepdims=True),uc-uc.mean(1,keepdims=True),fc-fc.mean(1,keepdims=True)]
        out[start:end]=np.stack(arrays,axis=2)
    return out.reshape(-1,len(names)),names

def context_features(data,indices,cutoff):
    stats=context_stats(cutoff)
    items=np.array(data["items"][indices]);users=np.array(data["users"][indices])
    ts=np.array(data["timestamps"][indices]);bucket=((ts%86400)//21600).astype(np.int32)
    weekend=np.array(data["context"][indices,2]).astype(np.int32)
    arrays=[];names=[]
    for name,ids in (("user",users),("item",items)):
        for label,cond in (("bucket",bucket),("weekend",weekend)):
            values=stats[name+"_"+label+"_affinity"][ids,cond if name=="user" else cond[:,None]]
            if name=="user":values=np.broadcast_to(values[:,None],items.shape)
            arrays.append(values);names.append(name+"_"+label+"_affinity")
        values=stats[name+"_bucket_ctr_offset"][ids,bucket if name=="user" else bucket[:,None]]
        if name=="user":values=np.broadcast_to(values[:,None],items.shape)
        arrays.append(values);names.append(name+"_bucket_ctr_offset")
    h=np.array(data["history"][indices]);age=np.array(data["age_hours"][indices])
    hminutes=ts[:,None]//60-np.rint(age*60).astype(np.int64)
    same=(hminutes%1440)//360==bucket[:,None]
    valid=h!=0
    arrays.append((same&valid).sum(1)[:,None].repeat(10,axis=1)/20);names.append("history_same_bucket_fraction")
    match=(items[:,:,None]==h[:,None,:])&valid[:,None,:]&same[:,None,:]
    arrays.append(match.sum(2));names.append("repeat_same_bucket_count")
    arrays.append(np.log1p(ts%86400/3600)[:,None].repeat(10,axis=1));names.append("hour_log")
    return np.stack(arrays,axis=2).reshape(-1,len(names)).astype(np.float32),names

def embedding_features(data,indices,cutoff,factors=16,seed=42,epochs=None,recipe=None):
    if recipe:seed=recipe.get('seed',seed)
    model=embedding_model(cutoff,factors,seed,epochs,recipe)
    users=np.array(data["users"][indices]);items=np.array(data["items"][indices]);history=np.array(data["history"][indices])
    ages=np.array(data["age_hours"][indices]);n=len(indices)
    names=[f"sgns{factors}_s{seed}_{s}" for s in ("dot","cos","usernorm","itemnorm","item_bias","hist_cos_mean",
        "hist_cos_max","hist_cos_recent","dot_relative","cos_relative","hist_same_bucket_cos","hist_same_weekend_cos")]
    out=np.empty((n,10,len(names)),dtype=np.float32)
    for start in range(0,n,4096):
        end=min(n,start+4096);u=model["users"][users[start:end]];v=model["items"][items[start:end]]
        un=np.linalg.norm(u,axis=1).clip(1e-6);vn=np.linalg.norm(v,axis=2).clip(1e-6)
        dot=(v*u[:,None,:]).sum(2);cos=dot/(un[:,None]*vn)
        h=model["items"][history[start:end]];hn=np.linalg.norm(h,axis=2).clip(1e-6)
        valid=history[start:end]!=0;hc=np.einsum('bik,bjk->bij',v,h)/(vn[:,:,None]*hn[:,None,:]);hc*=valid[:,None,:]
        length=valid.sum(1).clip(1);w=np.exp(-ages[start:end]/72)*valid
        ts=np.array(data["timestamps"][indices[start:end]])
        past=ts[:,None]//60-np.rint(ages[start:end]*60).astype(np.int64)
        same_bucket=((past%1440)//360==(ts%86400//21600)[:,None])&valid
        same_day=((((past//1440+3)%7)>=5)==(np.array(data["context"][indices[start:end],2])>0)[:,None])&valid
        arrays=[dot,cos,np.broadcast_to(un[:,None],dot.shape),vn,model["bias"][items[start:end]],
                hc.sum(2)/length[:,None],hc.max(2),(hc*w[:,None,:]).sum(2)/w.sum(1).clip(1e-6)[:,None],
                dot-dot.mean(1,keepdims=True),cos-cos.mean(1,keepdims=True),
                (hc*same_bucket[:,None,:]).sum(2)/same_bucket.sum(1).clip(1)[:,None],
                (hc*same_day[:,None,:]).sum(2)/same_day.sum(1).clip(1)[:,None]]
        out[start:end]=np.stack(arrays,axis=2)
    return out.reshape(-1,len(names)),names

def describe(rep, data, indices):
    blocks = []
    for offset in range(0, len(indices), 2048):
        ix = indices[offset:offset+2048]
        ids, hist = np.array(data['items'][ix]), np.array(data['history'][ix])
        age = np.array(data['age_hours'][ix])
        v, h, u = rep['items'][ids], rep['items'][hist], rep['users'][data['users'][ix]]
        vn = np.linalg.norm(v, axis=2).clip(1e-6)
        hn = np.linalg.norm(h, axis=2).clip(1e-6)
        un = np.linalg.norm(u, axis=1).clip(1e-6)
        mask = (hist != 0) & (np.linalg.norm(h, axis=2) > 0)
        dot = np.einsum('bid,bjd->bij', v, h)
        cos = dot/(vn[:, :, None]*hn[:, None, :])
        rank, length = np.cumsum(mask, axis=1), mask.sum(1)
        arrays, names = [], []
        def add(name, x, relative=False):
            if x.ndim == 1:
                x = np.broadcast_to(x[:, None], ids.shape)
            arrays.append(x)
            names.append('coclick_'+name)
            if relative:
                arrays.append(x-x.mean(1, keepdims=True))
                names.append('coclick_'+name+'_relative')
        def mean(weight):
            return (cos*weight[:, None, :]).sum(2)/weight.sum(1).clip(1e-6)[:, None]
        add('profile_dot', np.einsum('bid,bd->bi', v, u), True)
        add('profile_cos', np.einsum('bid,bd->bi', v, u)/(vn*un[:, None]), True)
        add('hist_mean', mean(mask), True)
        for k in [1, 3, 5]:
            add('last'+str(k), mean(mask & (rank > (length-k)[:, None])), True)
        for tau in [6, 24, 168]:
            add('decay'+str(tau), mean(mask*np.exp(-age/tau)), True)
        maxcos = np.where(mask[:, None, :], cos, -np.inf).max(2)
        add('hist_max', np.where(length[:, None] > 0, maxcos, 0))
        add('item_norm', vn)
        add('profile_norm', un)
        add('item_bias', rep['bias'][ids])
        blocks.append(np.stack(arrays, axis=2).astype(np.float32))
    result = np.concatenate(blocks).reshape(-1, len(names))
    assert np.isfinite(result).all()
    return result, names

def cosine(rep,raw,ix):
    u=rep['users'][raw['users'][ix]].astype(np.float64)
    v=rep['items'][raw['items'][ix]].astype(np.float64)
    u/=np.linalg.norm(u,axis=1,keepdims=True).clip(1e-8)
    v/=np.linalg.norm(v,axis=2,keepdims=True).clip(1e-8)
    return np.einsum('bd,bid->bi',u,v)

def from_representations(raw,ix,old,new,long):
    a,names=describe(new,raw,ix);b,old_names=describe(old,raw,ix)
    assert names==old_names
    cn=cosine(new,raw,ix);co=cosine(old,raw,ix);cl=cosine(long,raw,ix)
    d=cn-co;e=cn-cl
    extra=[d,d-d.mean(1,keepdims=True),e,e-e.mean(1,keepdims=True)]
    extra_names=['cos_delta_old','cos_delta_old_relative','cos_delta_long','cos_delta_long_relative']
    for name in ['profile_cos','hist_mean','last3','decay24','hist_max']:
        k=names.index('coclick_'+name);extra.append((a[:,k]-b[:,k]).reshape(len(ix),10))
        extra_names.append(name+'_delta_old')
    x=np.concatenate([a,np.stack(extra,2).reshape(-1,len(extra))],1).astype('f')
    cols=[n.replace('coclick_','unit_consensus_',1) for n in names]+['unit_consensus_'+n for n in extra_names]
    assert x.shape==(len(ix)*10,31) and np.isfinite(x).all()
    return x,cols

def build_features(data, asset_dir, fresh_models):
    """Raw evaluation events -> [events*10, 323], in the frozen column order."""
    global STORE, ASSET_DIR
    ASSET_DIR = asset_dir
    with np.load(asset_dir/'representations.npz', allow_pickle=False) as archive:
        STORE = dict(archive)
    set_num_threads(6)
    ix = np.arange(len(data['users']))
    cutoff = '2025-01-27'
    x, names, _, _ = base_features(data, ix, asset_dir)
    pieces = [x]

    def add(values, columns):
        pieces.append(values)
        names.extend(columns)

    for size in [8, 16]:
        add(*latent_features(data, ix, cutoff, size))
    add(*context_features(data, ix, cutoff))
    for size, seed in [(8, 2026), (16, 42), (8, 42)]:
        add(*embedding_features(data, ix, cutoff, size, seed))
    for width in [16, 64]:
        spectral = rep('spectral'+str(width))
        for size in ([8, 16] if width == 16 else [32, 64]):
            view = dict(spectral, users=spectral['users'][:, :size], items=spectral['items'][:, :size])
            a, cols = describe(view, data, ix)
            add(a, [c.replace('coclick_', 'spectral'+str(size)+'_', 1) for c in cols])
    for epoch, model in fresh_models.items():
        a, cols = describe(model.representation(), data, ix)
        offset = np.repeat(model.user_offset()[data['users'][ix]], 10)[:, None]
        p = model.predict(data).ravel().astype(np.float64).clip(1e-7, 1-1e-7)
        log_odds = (np.log(p)-np.log1p(-p))[:, None]
        cols = [c.replace('coclick_', 'fresh'+str(epoch)+'_', 1) for c in cols]
        add(np.concatenate([a, offset, log_odds], 1).astype('f'),
            cols+['fresh'+str(epoch)+'_user_offset', 'fresh'+str(epoch)+'_event_logit'])
    for epoch in [64, 128]:
        a, cols = describe(rep('long'+str(epoch)), data, ix)
        add(a, [c.replace('coclick_', 'spherelong'+str(epoch)+'_', 1) for c in cols])
    add(*from_representations(data, ix, rep('old32'), rep('unit_consensus'), rep('long128')))
    result = np.concatenate(pieces, 1)
    if result.shape != (len(ix)*10, 323) or not np.isfinite(result).all():
        raise ValueError('Invalid V31 feature matrix')
    STORE = None
    return result, names
