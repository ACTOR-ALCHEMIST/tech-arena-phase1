"""Compact conditional-choice factors with a sampled full-catalog partition."""
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


def partition_loss(u, positive, positive_bias, negative, negative_bias, n_items, excluded=None):
    pos = (u*positive).sum(1)+positive_bias
    neg = u@negative.T+negative_bias[None]
    if excluded is not None:
        neg = neg.masked_fill(excluded, -torch.inf)
    # Shared negatives are uniform. Scale their sum to estimate the catalog Z.
    neg = neg+math.log((n_items-1)/len(negative))
    return (torch.logaddexp(pos, torch.logsumexp(neg, 1))-pos).mean()


def model(cutoff, epochs=90):
    from arena_sgns_features import embedding_model
    folder = snapshot(cutoff)/'conditional_softmax_v1_d8'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/('result_e'+str(epochs)+'.json')).exists():
            return dict(np.load(folder/('factors_e'+str(epochs)+'.npz')))
        initial = embedding_model(cutoff, 8, recipe={'epochs': 320, 'lr': .01, 'lr_decay': False,
                  'seed': 2026, 'reuse_experiment': 'sgns8_const_e320_s2026'})
        x = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
        au, ai = x.nonzero()
        torch.manual_seed(9281)
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        u = nn.Embedding.from_pretrained(torch.tensor(initial['users'], device=device), freeze=False, padding_idx=0)
        v = nn.Embedding.from_pretrained(torch.tensor(initial['items'], device=device), freeze=False, padding_idx=0)
        bias = nn.Embedding.from_pretrained(torch.tensor(initial['bias'][:, None], device=device), freeze=False, padding_idx=0)
        optimizer = torch.optim.AdamW(list(u.parameters())+list(v.parameters())+list(bias.parameters()), lr=.003, weight_decay=.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90, eta_min=.0005)
        au, ai = torch.tensor(au, device=device), torch.tensor(ai, device=device)
        records = []
        for epoch in range(1, 91):
            start = time.monotonic()
            order = torch.randperm(len(au), device=device)
            total = 0.
            for offset in range(0, len(order), 8192):
                take = order[offset:offset+8192]
                uid, iid = au[take], ai[take]
                nid = torch.randint(1, x.shape[1], (256,), device=device)
                uv, pv, nv = u(uid), v(iid), v(nid)
                loss = partition_loss(uv, pv, bias(iid).squeeze(1), nv, bias(nid).squeeze(1), x.shape[1], nid[None] == iid[:, None])
                loss = loss+.003*(uv.square().mean()+pv.square().mean()+nv.square().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(u.parameters())+list(v.parameters()), 5)
                optimizer.step()
                total += loss.item()*len(take)
            scheduler.step()
            row = dict(epoch=epoch, loss=total/len(au), seconds=time.monotonic()-start)
            records.append(row)
            if epoch%5 == 0:
                print(dict(stage='conditional_softmax', cutoff=cutoff, **row), flush=True)
            if epoch in [30, 90]:
                uf, vf = u.weight.detach().cpu().numpy().copy(), v.weight.detach().cpu().numpy().copy()
                uf[np.diff(x.indptr) == 0] = 0
                vf[np.asarray(x.sum(0)).ravel() == 0] = 0
                uf[0], vf[0] = 0, 0
                np.savez(folder/('factors_e'+str(epoch)+'.npz'), users=uf, items=vf, bias=bias.weight.detach().cpu().numpy().ravel())
                write_json(folder/('result_e'+str(epoch)+'.json'), dict(cutoff=cutoff, epochs=records.copy(),
                           fixed_epochs=epoch, positives=len(au), catalog_items=x.shape[1]-1, negatives=256, source='Past-only unique user-item clicks',
                           initialization='sgns8_s2026_e320', seed=9281, lr=.003, lr_schedule_epochs=90))
    return dict(np.load(folder/('factors_e'+str(epochs)+'.npz')))


def extra_features(data, indices, cutoff, config=None, epochs=90):
    from arena_coclick import describe
    x, names = describe(model(cutoff, epochs), data, indices)
    return x, [n.replace('coclick_', 'softmax'+str(epochs)+'_', 1) for n in names]
