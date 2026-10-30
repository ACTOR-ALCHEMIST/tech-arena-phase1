"""Past-only item co-click representations, separate from user-ID SGNS."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.55')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import fcntl
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from arena_core import snapshot
from arena_ood import write_json

RECIPE = dict(version=1, factors=16, window=5, epochs=30, negatives=8,
              batch_size=16384, seed=941, lr=.008, regularization=.003)


def pairs(clicks, window=5):
    """Unordered neighboring click pairs; never cross a user's boundary."""
    order = np.lexsort((clicks['minutes'], clicks['users']))
    u, i = clicks['users'][order], clicks['items'][order]
    left, right = [], []
    for gap in range(1, window+1):
        valid = (u[:-gap] == u[gap:]) & (i[:-gap] != i[gap:])
        left.append(i[:-gap][valid])
        right.append(i[gap:][valid])
    return np.concatenate(left).astype(np.int32), np.concatenate(right).astype(np.int32)


def model(cutoff):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from arena_feedback import feedback
    from arena_sgns_features import embedding_model
    from scipy import sparse
    folder = snapshot(cutoff)/'coclick_v1_d16_w5_e30'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            return dict(np.load(folder/'factors.npz'))
        start = time.monotonic()
        clicks = feedback(cutoff)
        a, b = pairs(clicks, RECIPE['window'])
        shape = sparse.load_npz(snapshot(cutoff)/'interactions.npz').shape
        torch.manual_seed(RECIPE['seed'])
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        embed = nn.Embedding(shape[1], RECIPE['factors'], padding_idx=0, device=device)
        bias = nn.Embedding(shape[1], 1, padding_idx=0, device=device)
        # Warm start only the common first eight coordinates. New coordinates
        # retain noise to allow the co-click objective to learn a distinct view.
        initial = embedding_model(cutoff, 8, recipe={'epochs': 320, 'lr': .01,
                  'lr_decay': False, 'seed': 2026, 'reuse_experiment': 'sgns8_const_e320_s2026'})
        with torch.no_grad():
            embed.weight.normal_(std=.08)
            embed.weight[:, :8].copy_(torch.tensor(initial['items']*.6, device=device))
            embed.weight[0].zero_()
            bias.weight.zero_()
        optimizer = torch.optim.AdamW(list(embed.parameters())+list(bias.parameters()), lr=RECIPE['lr'], weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RECIPE['epochs'], eta_min=.001)
        ta, tb = torch.tensor(a, device=device), torch.tensor(b, device=device)
        records = []
        for epoch in range(1, RECIPE['epochs']+1):
            tick = time.monotonic()
            order = torch.randperm(len(a), device=device)
            total = 0.
            for start_batch in range(0, len(a), RECIPE['batch_size']):
                take = order[start_batch:start_batch+RECIPE['batch_size']]
                x, y = ta[take], tb[take]
                # Randomize which end supplies context, including both directions.
                flip = torch.rand(len(take), device=device) < .5
                x, y = torch.where(flip, y, x), torch.where(flip, x, y)
                neg = torch.randint(1, shape[1], (len(take), RECIPE['negatives']), device=device)
                u, v, n = embed(x), embed(y), embed(neg)
                pos = (u*v).sum(1)+bias(y).squeeze(1)-math.log(RECIPE['negatives'])
                nz = (u[:, None]*n).sum(2)+bias(neg).squeeze(2)-math.log(RECIPE['negatives'])
                loss = (F.softplus(-pos).mean()+RECIPE['negatives']*F.softplus(nz).mean())/(1+RECIPE['negatives'])
                loss = loss+RECIPE['regularization']*(u.square().mean()+v.square().mean()+n.square().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()*len(take)
            scheduler.step()
            row = dict(epoch=epoch, loss=total/len(a), seconds=time.monotonic()-tick)
            records.append(row)
            print(json.dumps(dict(stage='coclick_embedding', cutoff=cutoff, pairs=len(a), **row)), flush=True)
        v = embed.weight.detach().cpu().numpy()
        known = np.bincount(clicks['items'], minlength=shape[1]) > 0
        v[~known] = 0
        v[0] = 0
        # Long-term profile uses only clicks observed before the snapshot cutoff.
        profile = np.zeros((shape[0], v.shape[1]), dtype=np.float32)
        np.add.at(profile, clicks['users'], v[clicks['items']])
        count = np.bincount(clicks['users'], minlength=shape[0])
        profile /= count.clip(1)[:, None]
        np.savez(folder/'factors.npz', items=v, users=profile, bias=bias.weight.detach().cpu().numpy().ravel())
        write_json(folder/'result.json', dict(cutoff=cutoff, recipe=RECIPE, clicks=len(clicks['items']), pairs=len(a),
                   max_click_minute=int(clicks['minutes'].max()), pairs_sha256=hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest(),
                   epochs=records, seconds=time.monotonic()-start))
    return dict(np.load(folder/'factors.npz'))


def extra_features(data, indices, cutoff, config=None):
    rep = model(cutoff)
    return describe(rep, data, indices)


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


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cutoff', default='2025-01-24')
    a = p.parse_args()
    model(a.cutoff)
