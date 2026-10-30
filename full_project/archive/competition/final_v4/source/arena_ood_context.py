"""Shrinkage-regularized personal time-bucket preferences in fixed SGNS space.

Only clicks observed before the requested snapshot train this auxiliary model.
Current-event labels are never inputs to feature extraction or profile fitting.
"""
import fcntl
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def context_model(cutoff, config):
    from arena_sgns_features import embedding_model
    from arena_core import snapshot
    from arena_feedback import feedback
    from arena_ood import write_json
    base = embedding_model(cutoff, 8, epochs=config['sgns_epochs']['8'], recipe=config['sgns_recipes']['8'])
    recipe = {'epochs': 24, 'seed': 73, 'lr': .005, 'anchor': .05, 'shrinkage': 8,
              'negatives': 8, 'version': 1, 'base_recipe': config['sgns_recipes']['8']}
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]
    folder = snapshot(cutoff)/('ood_context_'+digest)
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            return dict(np.load(folder/'factors.npz'))
        torch.manual_seed(recipe['seed'])
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        obs = feedback(cutoff)
        uid, iid = obs['users'], obs['items']
        bucket = (obs['minutes']%1440)//360
        nu, ni = len(base['users']), len(base['items'])
        support_u = np.bincount(uid.astype(np.int64)*4+bucket, minlength=nu*4).reshape(nu, 4).astype(np.float32)
        support_i = np.bincount(iid.astype(np.int64)*4+bucket, minlength=ni*4).reshape(ni, 4).astype(np.float32)
        shrink_u = torch.tensor(support_u/(support_u+recipe['shrinkage']), device=device)
        shrink_i = torch.tensor(support_i/(support_i+10), device=device)
        users = torch.tensor(base['users'], device=device)
        items = torch.tensor(base['items'], device=device)
        bias = torch.tensor(base['bias'], device=device)
        delta = nn.Embedding(nu*4, 8, padding_idx=0, device=device)
        contextual_bias = nn.Embedding(ni*4, 1, padding_idx=0, device=device)
        nn.init.zeros_(delta.weight)
        nn.init.zeros_(contextual_bias.weight)
        optimizer = torch.optim.AdamW(list(delta.parameters())+list(contextual_bias.parameters()), lr=recipe['lr'], weight_decay=0)
        ids_u = torch.tensor(uid, dtype=torch.long, device=device)
        ids_i = torch.tensor(iid, dtype=torch.long, device=device)
        ids_b = torch.tensor(bucket, dtype=torch.long, device=device)
        records = []
        start = time.monotonic()
        for epoch in range(1, recipe['epochs']+1):
            order = torch.randperm(len(uid), device=device)
            losses = []
            tick = time.monotonic()
            for offset in range(0, len(order), 16384):
                chosen = order[offset:offset+16384]
                u, i, b = ids_u[chosen], ids_i[chosen], ids_b[chosen]
                negative = torch.randint(1, ni, (len(chosen), recipe['negatives']), device=device)
                # Zero-sum bucket deviations preserve the global preference.
                raw_delta = delta(u*4+b)-delta.weight.reshape(nu, 4, 8)[u].mean(1)
                profile = users[u]+raw_delta*shrink_u[u, b, None]
                raw_pos = contextual_bias(i*4+b).squeeze(-1)-contextual_bias.weight.reshape(ni, 4)[i].mean(1)
                raw_neg = contextual_bias(negative*4+b[:, None]).squeeze(-1)-contextual_bias.weight.reshape(ni, 4)[negative].mean(2)
                pos = (profile*items[i]).sum(1)+bias[i]+raw_pos*shrink_i[i, b]-np.log(recipe['negatives'])
                neg = (profile[:, None, :]*items[negative]).sum(2)+bias[negative]+raw_neg*shrink_i[negative, b[:, None]]-np.log(recipe['negatives'])
                likelihood = (F.softplus(-pos).mean()+recipe['negatives']*F.softplus(neg).mean())/(recipe['negatives']+1)
                penalty = recipe['anchor']*(raw_delta.square().mean()+raw_pos.square().mean()+raw_neg.square().mean())
                loss = likelihood+penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            row = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=time.monotonic()-tick)
            records.append(row)
            if epoch == 1 or epoch%4 == 0:
                print(json.dumps(dict(stage='context_preferences', cutoff=cutoff, **row)), flush=True)
            write_json(folder/'training.json', records)
        d = delta.weight.detach().cpu().numpy().reshape(nu, 4, 8)
        d = (d-d.mean(1, keepdims=True))*(support_u/(support_u+recipe['shrinkage']))[..., None]
        b = contextual_bias.weight.detach().cpu().numpy().reshape(ni, 4)
        b = (b-b.mean(1, keepdims=True))*(support_i/(support_i+10))
        d[0] = 0
        b[0] = 0
        np.savez(folder/'factors.npz', delta=d, bias=b, user_support=support_u, item_support=support_i)
        write_json(folder/'result.json', dict(recipe=recipe, cutoff=cutoff, observed_clicks=len(uid),
                   max_click_timestamp=int(obs['minutes'].max())*60, seconds=time.monotonic()-start))
        return dict(np.load(folder/'factors.npz'))


def extra_features(data, indices, cutoff, config):
    from arena_sgns_features import embedding_model
    from arena_core import snapshot
    base = embedding_model(cutoff, 8, epochs=config['sgns_epochs']['8'], recipe=config['sgns_recipes']['8'])
    context = context_model(cutoff, config)
    known = np.load(snapshot(cutoff)/'stats.npz')['known_users']
    blocks = []
    names = ['context8_'+x for x in ['dot', 'cos', 'dot_relative', 'cos_relative', 'delta_dot', 'delta_cos',
              'delta_norm', 'profile_norm', 'item_bucket_bias', 'user_bucket_support', 'item_bucket_support', 'global_context_cos']]
    for start in range(0, len(indices), 4096):
        ix = indices[start:start+4096]
        user_ids, item_ids = data['users'][ix], data['items'][ix]
        bucket = ((data['timestamps'][ix]%86400)//21600).astype(np.int32)
        u = base['users'][user_ids].copy()
        u[~known[user_ids]] = 0
        d = context['delta'][user_ids, bucket]
        profile = u+d
        v = base['items'][item_ids]
        un, dn, pn = [np.linalg.norm(a, axis=1).clip(1e-6) for a in [u, d, profile]]
        vn = np.linalg.norm(v, axis=2).clip(1e-6)
        dot = (profile[:, None, :]*v).sum(2)
        cosine = dot/(pn[:, None]*vn)
        delta_dot = (d[:, None, :]*v).sum(2)
        event_shape = dot.shape
        values = [dot, cosine, dot-dot.mean(1, keepdims=True), cosine-cosine.mean(1, keepdims=True),
                  delta_dot, delta_dot/(dn[:, None]*vn), np.broadcast_to(dn[:, None], event_shape),
                  np.broadcast_to(pn[:, None], event_shape), context['bias'][item_ids, bucket[:, None]],
                  np.broadcast_to(np.log1p(context['user_support'][user_ids, bucket])[:, None], event_shape),
                  np.log1p(context['item_support'][item_ids, bucket[:, None]]),
                  np.broadcast_to(((u*profile).sum(1)/(un*pn))[:, None], event_shape)]
        blocks.append(np.stack(values, axis=2).astype(np.float32))
    result = np.concatenate(blocks).reshape(-1, len(names))
    assert np.isfinite(result).all()
    return result, names
