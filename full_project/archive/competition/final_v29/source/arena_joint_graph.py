"""Joint user/item click factors and tied item co-click geometry."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.55')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import fcntl
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import sparse
from arena_core import snapshot
from arena_ood import write_json


def model(cutoff):
    from arena_sgns_features import embedding_model
    from arena_feedback import feedback
    from arena_coclick import pairs
    folder = snapshot(cutoff)/'joint_graph_v1_d8_e100'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            return dict(np.load(folder/'factors.npz'))
        initial = embedding_model(cutoff, 8, recipe={'epochs': 320, 'lr': .01, 'lr_decay': False,
                  'seed': 2026, 'reuse_experiment': 'sgns8_const_e320_s2026'})
        x = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
        au, ai = x.nonzero()
        left, right = pairs(feedback(cutoff), 5)
        known = np.asarray(x.sum(0)).ravel() > 0
        item = initial['items'].astype(np.float64)
        center = item[known].mean(0)
        centered = item-center
        covariance = centered[known].T@centered[known]/known.sum()
        value, vectors = np.linalg.eigh(covariance)
        transform = (vectors*(.6/np.sqrt(value+.1*value.mean())))@vectors.T
        v0 = (centered@transform).astype('f')
        u0 = (initial['users']@np.linalg.inv(transform).T).astype('f')
        bu0 = (initial['users']@center).astype('f')
        v0[0], u0[0], bu0[0] = 0, 0, 0
        torch.manual_seed(8321)
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        u = nn.Embedding.from_pretrained(torch.tensor(u0, device=device), freeze=False, padding_idx=0)
        v = nn.Embedding.from_pretrained(torch.tensor(v0, device=device), freeze=False, padding_idx=0)
        bu = nn.Embedding.from_pretrained(torch.tensor(bu0[:, None], device=device), freeze=False, padding_idx=0)
        bi = nn.Embedding.from_pretrained(torch.tensor(initial['bias'][:, None], device=device), freeze=False, padding_idx=0)
        cb = nn.Embedding(x.shape[1], 1, padding_idx=0, device=device)
        nn.init.zeros_(cb.weight)
        log_alpha = nn.Parameter(torch.tensor(math.log(.25), device=device))
        params = list(u.parameters())+list(v.parameters())+list(bu.parameters())+list(bi.parameters())+list(cb.parameters())+[log_alpha]
        optimizer = torch.optim.AdamW(params, lr=.005, weight_decay=.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=.001)
        t_u0, t_v0 = torch.tensor(u0, device=device), torch.tensor(v0, device=device)
        au, ai = torch.tensor(au, device=device), torch.tensor(ai, device=device)
        left, right = torch.tensor(left, device=device), torch.tensor(right, device=device)
        records = []
        def nce(pos, neg):
            return (F.softplus(-pos+math.log(8)).mean()+8*F.softplus(neg-math.log(8)).mean())/9
        for epoch in range(1, 101):
            start = time.monotonic()
            total = 0.
            order = torch.randperm(len(au), device=device)
            for offset in range(0, len(order), 8192):
                take = order[offset:offset+8192]
                uid, iid = au[take], ai[take]
                ni = torch.randint(1, x.shape[1], (len(take), 8), device=device)
                uv, vv, nv = u(uid), v(iid), v(ni)
                ui_pos = (uv*vv).sum(1)+bi(iid).squeeze(1)+bu(uid).squeeze(1)
                ui_neg = (uv[:, None]*nv).sum(2)+bi(ni).squeeze(2)+bu(uid)
                pair = torch.randint(len(left), (len(take),), device=device)
                a, b = left[pair], right[pair]
                flip = torch.rand(len(take), device=device) < .5
                a, b = torch.where(flip, b, a), torch.where(flip, a, b)
                av, bv = v(a), v(b)
                scale = log_alpha.exp().clamp(.03, 2)
                co_pos = scale*(av*bv).sum(1)+bi(b).squeeze(1)+cb(a).squeeze(1)
                co_neg = scale*(av[:, None]*nv).sum(2)+bi(ni).squeeze(2)+cb(a)
                loss = nce(ui_pos, ui_neg)+.25*nce(co_pos, co_neg)
                loss = loss+.01*((uv-t_u0[uid]).square().mean()+(vv-t_v0[iid]).square().mean())
                loss = loss+.001*(uv.square().mean()+vv.square().mean()+nv.square().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()*len(take)
            scheduler.step()
            row = dict(epoch=epoch, loss=total/len(au), co_scale=float(log_alpha.exp().item()), seconds=time.monotonic()-start)
            records.append(row)
            if epoch%10 == 0:
                print(dict(stage='joint_graph', cutoff=cutoff, **row), flush=True)
        uf, vf = u.weight.detach().cpu().numpy().copy(), v.weight.detach().cpu().numpy().copy()
        uf[np.diff(x.indptr) == 0] = 0
        vf[~known] = 0
        uf[0], vf[0] = 0, 0
        np.savez(folder/'factors.npz', users=uf, items=vf, bias=bi.weight.detach().cpu().numpy().ravel(), user_bias=bu.weight.detach().cpu().numpy().ravel())
        write_json(folder/'result.json', dict(cutoff=cutoff, epochs=records, factors=8, co_weight=.25, seed=8321,
                   anchor=.01, learning_rate=.005, lr_schedule_epochs=100, negatives=8, click_pairs=len(left), unique_user_item=len(au)))
    return dict(np.load(folder/'factors.npz'))


def extra_features(data, indices, cutoff, config=None):
    from arena_coclick import describe
    rep = model(cutoff)
    x, names = describe(rep, data, indices)
    bias = np.repeat(rep['user_bias'][data['users'][indices]], 10)[:, None]
    return np.concatenate([x, bias], 1), [n.replace('coclick_', 'joint_graph_', 1) for n in names]+['joint_graph_user_bias']
