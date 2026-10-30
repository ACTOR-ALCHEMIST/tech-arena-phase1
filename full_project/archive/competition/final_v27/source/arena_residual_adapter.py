"""Low-rank preference adapters on immutable SGNS8, initialized by the click graph."""
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


def initialize_graph_residual(base, graph, known, rank):
    """Remove graph directions linearly predictable from the existing item frame."""
    original = base['items'].astype(np.float64)
    g = graph['items'][:, :max(rank, 16)].astype(np.float64)
    design = np.c_[original[known], np.ones(known.sum())]
    fit = np.linalg.lstsq(design, g[known], rcond=None)[0]
    residual = g-np.c_[original, np.ones(len(original))]@fit
    _, _, axes = np.linalg.svd(residual[known], full_matrices=False)
    v = residual@axes[:rank].T
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(1e-8)
    v *= .7
    v[~known], v[0] = 0, 0
    return v.astype('f')


def model(cutoff, rank=8, graph_only=False):
    from arena_sgns_features import embedding_model
    from arena_spectral import model as spectral
    mode = 'frozen_graph16' if graph_only else 'residual'+str(rank)
    folder = snapshot(cutoff)/('adapter_v1_'+mode+'_e160')
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            return dict(np.load(folder/'factors.npz'))
        base = embedding_model(cutoff, 8, recipe={'epochs': 320, 'lr': .01, 'lr_decay': False,
                  'seed': 2026, 'reuse_experiment': 'sgns8_const_e320_s2026'})
        x = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
        known = np.asarray(x.sum(0)).ravel() > 0
        degree = np.diff(x.indptr)
        graph = spectral(cutoff, 16 if graph_only or rank <= 16 else 64)
        if graph_only:
            v0 = graph['items'][:, :16].copy()
            v0 /= np.linalg.norm(v0, axis=1, keepdims=True).clip(1e-8)
            v0 *= 2
            u0 = (x@v0/degree.clip(1)[:, None]).astype('f')
            extra_rank = 16
        else:
            v0 = initialize_graph_residual(base, graph, known, rank)
            u0 = np.zeros((x.shape[0], rank), 'f')
            extra_rank = rank
        v0[~known], v0[0], u0[0] = 0, 0, 0
        torch.manual_seed(9101+extra_rank)
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        users = nn.Embedding.from_pretrained(torch.tensor(u0, device=device), freeze=False, padding_idx=0)
        items = nn.Embedding.from_pretrained(torch.tensor(v0, device=device), freeze=graph_only, padding_idx=0)
        bu = nn.Embedding(x.shape[0], 1, padding_idx=0, device=device)
        bi = nn.Embedding.from_pretrained(torch.tensor(base['bias'][:, None], device=device), freeze=False, padding_idx=0)
        nn.init.zeros_(bu.weight)
        fixed_u, fixed_v = torch.tensor(base['users'], device=device), torch.tensor(base['items'], device=device)
        anchor_v = torch.tensor(v0, device=device)
        pop = np.asarray(x.sum(0)).ravel()
        shrink = torch.tensor((20/(pop+20)).astype('f'), device=device)
        parameters = list(users.parameters())+list(bu.parameters())+list(bi.parameters())
        if not graph_only:
            parameters += list(items.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=.006, weight_decay=.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160, eta_min=.001)
        au, ai = x.nonzero()
        au, ai = torch.tensor(au, device=device), torch.tensor(ai, device=device)
        records = []
        for epoch in range(1, 161):
            start = time.monotonic()
            total = 0.
            order = torch.randperm(len(au), device=device)
            for offset in range(0, len(order), 8192):
                take = order[offset:offset+8192]
                uid, iid = au[take], ai[take]
                ni = torch.randint(1, x.shape[1], (len(take), 8), device=device)
                u, v, n = users(uid), items(iid), items(ni)
                pos = (u*v).sum(1)
                neg = (u[:, None]*n).sum(2)
                if not graph_only:
                    pos = pos+(fixed_u[uid]*fixed_v[iid]).sum(1)
                    neg = neg+(fixed_u[uid, None]*fixed_v[ni]).sum(2)
                pos = pos+bi(iid).squeeze(1)+bu(uid).squeeze(1)-math.log(8)
                neg = neg+bi(ni).squeeze(2)+bu(uid)-math.log(8)
                loss = (F.softplus(-pos).mean()+8*F.softplus(neg).mean())/9
                loss = loss+(.01 if graph_only else .03)*u.square().mean()
                if not graph_only:
                    loss = loss+.05*((v-anchor_v[iid]).square().mean(1)*shrink[iid]).mean()
                    loss = loss+.05*((n-anchor_v[ni]).square().mean(2)*shrink[ni]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()*len(take)
            scheduler.step()
            row = dict(epoch=epoch, loss=total/len(au), seconds=time.monotonic()-start)
            records.append(row)
            if epoch%20 == 0:
                print(dict(stage='graph_preference_adapter', cutoff=cutoff, mode=mode, **row), flush=True)
        u, v = users.weight.detach().cpu().numpy(), items.weight.detach().cpu().numpy()
        if not graph_only:
            u = np.concatenate([base['users'], u], 1)
            v = np.concatenate([base['items'], v], 1)
        u[degree == 0], v[~known], u[0], v[0] = 0, 0, 0, 0
        np.savez(folder/'factors.npz', users=u, items=v, bias=bi.weight.detach().cpu().numpy().ravel(),
                 user_bias=bu.weight.detach().cpu().numpy().ravel())
        write_json(folder/'result.json', dict(cutoff=cutoff, mode=mode, adapter_rank=extra_rank, epochs=records,
                   base_frozen=not graph_only, item_frozen=graph_only, unique_user_item=len(au),
                   seed=9101+extra_rank, lr=.006, schedule_epochs=160, item_anchor=.05, negatives=8))
    return dict(np.load(folder/'factors.npz'))


def extra_features(data, indices, cutoff, config=None, mode='residual8'):
    from arena_coclick import describe
    rep = model(cutoff, int(mode[8:]) if mode.startswith('residual') else 8, mode == 'frozen_graph16')
    x, names = describe(rep, data, indices)
    bias = np.repeat(rep['user_bias'][data['users'][indices]], 10)[:, None]
    return np.concatenate([x, bias], 1), [n.replace('coclick_', 'adapter_'+mode+'_', 1) for n in names]+['adapter_'+mode+'_user_bias']
