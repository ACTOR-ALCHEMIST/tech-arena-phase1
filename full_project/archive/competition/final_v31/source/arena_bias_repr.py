"""User-intercept NCE: keep all eight preference coordinates for interactions."""
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


def model(cutoff, epochs=240):
    from arena_sgns_features import embedding_model
    folder = snapshot(cutoff)/'nce_user_bias_v1_d8'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/('result_e'+str(epochs)+'.json')).exists():
            return dict(np.load(folder/('factors_e'+str(epochs)+'.npz')))
        initial = embedding_model(cutoff, 8, recipe={'epochs': 320, 'lr': .01, 'lr_decay': False,
                  'seed': 2026, 'reuse_experiment': 'sgns8_const_e320_s2026'})
        x = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
        au, ai = x.nonzero()
        torch.manual_seed(8329)
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        u = nn.Embedding.from_pretrained(torch.tensor(initial['users'], device=device), freeze=False, padding_idx=0)
        v = nn.Embedding.from_pretrained(torch.tensor(initial['items'], device=device), freeze=False, padding_idx=0)
        bi = nn.Embedding.from_pretrained(torch.tensor(initial['bias'][:, None], device=device), freeze=False, padding_idx=0)
        bu = nn.Embedding(x.shape[0], 1, device=device, padding_idx=0)
        nn.init.zeros_(bu.weight)
        optimizer = torch.optim.AdamW([dict(params=list(u.parameters())+list(v.parameters())+list(bi.parameters()), lr=.005, weight_decay=.01),
                                       dict(params=bu.parameters(), lr=.02, weight_decay=.001)])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=240, eta_min=.001)
        au, ai = torch.tensor(au, device=device), torch.tensor(ai, device=device)
        records = []
        for epoch in range(1, 241):
            start = time.monotonic()
            order = torch.randperm(len(au), device=device)
            total = 0.
            for offset in range(0, len(order), 8192):
                take = order[offset:offset+8192]
                uid, iid = au[take], ai[take]
                ni = torch.randint(1, x.shape[1], (len(take), 8), device=device)
                uv, vv, nv = u(uid), v(iid), v(ni)
                pos = (uv*vv).sum(1)+bi(iid).squeeze(1)+bu(uid).squeeze(1)-math.log(8)
                neg = (uv[:, None]*nv).sum(2)+bi(ni).squeeze(2)+bu(uid)-math.log(8)
                loss = (F.softplus(-pos).mean()+8*F.softplus(neg).mean())/9
                loss = loss+.001*(uv.square().mean()+vv.square().mean()+nv.square().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()*len(take)
            scheduler.step()
            row = dict(epoch=epoch, loss=total/len(au), seconds=time.monotonic()-start)
            records.append(row)
            if epoch%20 == 0:
                print(dict(stage='nce_user_bias', cutoff=cutoff, **row), flush=True)
            if epoch in [80, 240]:
                uf, vf = u.weight.detach().cpu().numpy().copy(), v.weight.detach().cpu().numpy().copy()
                uf[np.diff(x.indptr) == 0] = 0
                vf[np.asarray(x.sum(0)).ravel() == 0] = 0
                uf[0], vf[0] = 0, 0
                np.savez(folder/('factors_e'+str(epoch)+'.npz'), users=uf, items=vf,
                         bias=bi.weight.detach().cpu().numpy().ravel(), user_bias=bu.weight.detach().cpu().numpy().ravel())
                write_json(folder/('result_e'+str(epoch)+'.json'), dict(cutoff=cutoff, epochs=records.copy(), fixed_epochs=epoch,
                           factors=8, unique_user_item=len(au), negatives=8, seed=8329, embedding_lr=.005, user_bias_lr=.02,
                           lr_schedule_epochs=240, initialization='sgns8_s2026_e320'))
    return dict(np.load(folder/('factors_e'+str(epochs)+'.npz')))


def extra_features(data, indices, cutoff, config=None, epochs=240):
    from arena_coclick import describe
    rep = model(cutoff, epochs)
    x, names = describe(rep, data, indices)
    bias = np.repeat(rep['user_bias'][data['users'][indices]], 10)[:, None]
    tag = 'nce_bias'+str(epochs)+'_'
    return np.concatenate([x, bias], 1), [n.replace('coclick_', tag, 1) for n in names]+[tag+'user_bias']
