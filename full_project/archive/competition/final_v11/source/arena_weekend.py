"""Hierarchical weekday/weekend preferences learned only from prior clicks."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.55')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.4')
import fcntl
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_core import snapshot
from arena_ood import write_json


def model(cutoff, config):
    from arena_sgns_features import embedding_model
    from arena_feedback import feedback
    base = embedding_model(cutoff, 8, recipe=config['sgns_recipes']['8'])
    from scipy import sparse
    interactions = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
    known_u = np.diff(interactions.indptr) > 0
    known_i = np.asarray(interactions.sum(0)).ravel() > 0
    base['users'][~known_u] = 0
    base['items'][~known_i] = 0
    folder = snapshot(cutoff)/'weekday_weekend_v1_d8_e32'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            return dict(np.load(folder/'factors.npz'))
        torch.manual_seed(8292)
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        obs = feedback(cutoff)
        uid, iid = obs['users'], obs['items']
        # Unix epoch Thursday: weekday = (days + 3) % 7.
        kind = (((obs['minutes']//1440+3)%7) >= 5).astype(np.int32)
        nu, ni = len(base['users']), len(base['items'])
        su = np.bincount(uid.astype(np.int64)*2+kind, minlength=nu*2).reshape(nu, 2).astype('f')
        si = np.bincount(iid.astype(np.int64)*2+kind, minlength=ni*2).reshape(ni, 2).astype('f')
        shrink_u, shrink_i = torch.tensor(su/(su+6), device=device), torch.tensor(si/(si+10), device=device)
        u, v, b = [torch.tensor(base[k], device=device) for k in ['users', 'items', 'bias']]
        delta = nn.Embedding(nu*2, 8, padding_idx=0, device=device)
        ib = nn.Embedding(ni*2, 1, padding_idx=0, device=device)
        intercept = nn.Embedding(nu*2, 1, padding_idx=0, device=device)
        for layer in [delta, ib, intercept]:
            nn.init.zeros_(layer.weight)
        # Shared maps borrow evidence across users with similar global profiles.
        transform = nn.Parameter(torch.zeros(2, 8, 8, device=device))
        global_delta = nn.Parameter(torch.zeros(2, 8, device=device))
        params = list(delta.parameters())+list(ib.parameters())+list(intercept.parameters())+[transform, global_delta]
        optimizer = torch.optim.AdamW(params, lr=.004, weight_decay=.001)
        tu, ti, tk = [torch.tensor(a, dtype=torch.long, device=device) for a in [uid, iid, kind]]
        records = []
        for epoch in range(1, 33):
            start = time.monotonic()
            order = torch.randperm(len(uid), device=device)
            total = 0.
            for offset in range(0, len(order), 16384):
                chosen = order[offset:offset+16384]
                a, c, k = tu[chosen], ti[chosen], tk[chosen]
                n = torch.randint(1, ni, (len(chosen), 8), device=device)
                d = delta(a*2+k)-delta.weight.reshape(nu, 2, 8)[a].mean(1)
                t = transform[k]-transform.mean(0)
                gd = global_delta[k]-global_delta.mean(0)
                profile = u[a]+torch.bmm(u[a, None], t).squeeze(1)+gd+d*shrink_u[a, k, None]
                pb = ib(c*2+k).squeeze(1)-ib.weight.reshape(ni, 2)[c].mean(1)
                nb = ib(n*2+k[:, None]).squeeze(2)-ib.weight.reshape(ni, 2)[n].mean(2)
                offset_bias = intercept(a*2+k)
                pos = (profile*v[c]).sum(1)+b[c]+pb*shrink_i[c, k]+offset_bias.squeeze(1)-np.log(8)
                neg = (profile[:, None]*v[n]).sum(2)+b[n]+nb*shrink_i[n, k[:, None]]+offset_bias-np.log(8)
                loss = (F.softplus(-pos).mean()+8*F.softplus(neg).mean())/9
                loss = loss+.1*d.square().mean()+.05*(pb.square().mean()+nb.square().mean())+.02*(t.square().mean()+gd.square().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()*len(chosen)
            row = dict(epoch=epoch, loss=total/len(uid), seconds=time.monotonic()-start)
            records.append(row)
            if epoch%8 == 0:
                print(dict(stage='weekday_weekend', cutoff=cutoff, **row), flush=True)
        d = delta.weight.detach().cpu().numpy().reshape(nu, 2, 8)
        d = (d-d.mean(1, keepdims=True))*(su/(su+6))[..., None]
        d[0] = 0
        t, gd = transform.detach().cpu().numpy(), global_delta.detach().cpu().numpy()
        t, gd = t-t.mean(0), gd-gd.mean(0)
        contextual = base['users'][:, None]+np.einsum('ud,kde->uke', base['users'], t)+gd[None]+d
        item_bias = ib.weight.detach().cpu().numpy().reshape(ni, 2)
        item_bias = (item_bias-item_bias.mean(1, keepdims=True))*(si/(si+10))
        item_bias[0] = 0
        np.savez(folder/'factors.npz', profiles=contextual.astype('f'), bias=item_bias,
                 user_support=su, item_support=si, transforms=t, global_delta=gd, known_users=known_u, known_items=known_i)
        write_json(folder/'result.json', dict(cutoff=cutoff, base_recipe=config['sgns_recipes']['8'], epochs=records, seed=8292, personal_anchor=.1, shared_anchor=.02,
                   user_shrinkage=6, item_shrinkage=10, observed_clicks=len(uid), max_click_timestamp=int(obs['minutes'].max())*60,
                   source='Past-only distinct click timestamps; shared context maps plus shrunken personal deviations'))
    return dict(np.load(folder/'factors.npz'))


def extra_features(data, indices, cutoff, config):
    from arena_sgns_features import embedding_model
    base = embedding_model(cutoff, 8, recipe=config['sgns_recipes']['8'])
    context = model(cutoff, config)
    names = ['weekend_'+n for n in ['dot', 'cos', 'dot_relative', 'cos_relative', 'delta_dot', 'delta_cos',
              'delta_norm', 'profile_norm', 'item_type_bias', 'user_type_support', 'item_type_support', 'global_context_cos']]
    pieces = []
    for start in range(0, len(indices), 4096):
        ix = indices[start:start+4096]
        uid, iid, k = data['users'][ix], data['items'][ix], (data['context'][ix, 2] > 0).astype(np.int32)
        u = base['users'][uid].copy()
        u[~context['known_users'][uid]] = 0
        profile = context['profiles'][uid, k]
        delta = profile-u
        v = base['items'][iid].copy()
        v[~context['known_items'][iid]] = 0
        un, dn, pn = [np.linalg.norm(a, axis=1).clip(1e-6) for a in [u, delta, profile]]
        vn = np.linalg.norm(v, axis=2).clip(1e-6)
        dot = np.einsum('bid,bd->bi', v, profile)
        cos = dot/(pn[:, None]*vn)
        delta_dot = np.einsum('bid,bd->bi', v, delta)
        expand = lambda x: np.broadcast_to(x[:, None], dot.shape)
        values = [dot, cos, dot-dot.mean(1, keepdims=True), cos-cos.mean(1, keepdims=True), delta_dot, delta_dot/(dn[:, None]*vn),
                  expand(dn), expand(pn), context['bias'][iid, k[:, None]], expand(np.log1p(context['user_support'][uid, k])),
                  np.log1p(context['item_support'][iid, k[:, None]]), expand((u*profile).sum(1)/(un*pn))]
        pieces.append(np.stack(values, 2).astype('f'))
    x = np.concatenate(pieces).reshape(-1, len(names))
    assert np.isfinite(x).all()
    return x, names
