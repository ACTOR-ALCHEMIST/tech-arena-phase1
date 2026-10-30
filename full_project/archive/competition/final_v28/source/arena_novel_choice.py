"""Refine latent preferences from actual, non-repeat clicked choice sets.

No-click events do not constrain conditional preference. Repeat candidates
are removed from the conditional set, to reduce history-injection shortcuts.
All fitting labels and histories precede the representation snapshot.
"""
import fcntl
import json
import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_ood import ROOT, write_json


def examples(data, cutoff):
    from run import timestamp
    indices = np.flatnonzero(data['timestamps'] < timestamp(cutoff))
    y = np.asarray(data['labels'][indices])
    clicked = y.sum(1) > 0
    indices, y = indices[clicked], y[clicked]
    targets = y.argmax(1)
    items, hist = data['items'][indices], data['history'][indices]
    repeat = ((items[:, :, None] == hist[:, None]) & (hist[:, None] != 0)).any(2)
    eligible = ~repeat[np.arange(len(indices)), targets]
    return indices[eligible], (~repeat[eligible]), targets[eligible]


class ChoiceFactors(nn.Module):
    def __init__(self, initial):
        super().__init__()
        u, v = torch.as_tensor(initial['users']), torch.as_tensor(initial['items'])
        self.users = nn.Embedding.from_pretrained(u.clone(), freeze=False, padding_idx=0)
        self.items = nn.Embedding.from_pretrained(v.clone(), freeze=False, padding_idx=0)
        self.register_buffer('u0', u.clone())
        self.register_buffer('v0', v.clone())
        self.register_buffer('bias', torch.as_tensor(initial['bias']))
        self.coefficients = nn.Parameter(torch.tensor([.25, .7, .5, .3]))

    def forward(self, uid, ids, hist, ages):
        u, v, h = self.users(uid), self.items(ids), self.items(hist)
        dot = (u[:, None]*v).sum(2)
        vnorm = v/torch.sqrt(1+v.square().sum(2, keepdim=True))
        hnorm = h/torch.sqrt(1+h.square().sum(2, keepdim=True))
        cos = torch.bmm(vnorm, hnorm.transpose(1, 2))
        visible = hist != 0
        weights = torch.exp(-ages/72)*visible
        mean = (cos*weights[:, None]).sum(2)/weights.sum(1).clamp_min(1e-6)[:, None]
        peak = cos.masked_fill(~visible[:, None], -1).max(2).values
        peak = torch.where(visible.any(1)[:, None], peak, 0)
        features = torch.stack([dot, self.bias[ids], mean, peak], 2)
        return (features*self.coefficients).sum(2)


def model(cutoff, epoch=40):
    from arena_core import snapshot
    from arena_sgns_features import embedding_model
    from ctr_data import load_cache
    from scipy import sparse
    folder = snapshot(cutoff)/'novel_choice_v1_d8_e40'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/('e'+str(epoch)+'.npz')).exists():
            return dict(np.load(folder/('e'+str(epoch)+'.npz')))
        initial = embedding_model(cutoff, 8, recipe=dict(epochs=320, lr=.01, lr_decay=False,
                                  seed=2026, reuse_experiment='sgns8_const_e320_s2026'))
        _, raw = load_cache(ROOT/'data/cache_v1', 'train')
        indices, allowed, targets = examples(raw, cutoff)
        x = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
        pu, pi = x.nonzero()
        torch.manual_seed(4187)
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        net = ChoiceFactors(initial).to(device)
        head_optimizer = torch.optim.AdamW([net.coefficients], lr=.003, weight_decay=.01)
        optimizer = torch.optim.AdamW([dict(params=[net.users.weight, net.items.weight], lr=.0005, weight_decay=0),
                                       dict(params=[net.coefficients], lr=.001, weight_decay=.01)])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=.0001)
        au, ai = torch.tensor(pu, device=device), torch.tensor(pi, device=device)
        rng, records = np.random.default_rng(4187), []
        for e in range(-7, 41):
            net.users.requires_grad_(e > 0)
            net.items.requires_grad_(e > 0)
            tick, total = time.monotonic(), 0.
            order = rng.permutation(len(indices))
            for start in range(0, len(order), 2048):
                pick = order[start:start+2048]
                ix = indices[pick]
                uid = torch.tensor(np.array(raw['users'][ix]), device=device)
                ids = torch.tensor(np.array(raw['items'][ix]), device=device)
                hist = torch.tensor(np.array(raw['history'][ix]), device=device)
                ages = torch.tensor(np.array(raw['age_hours'][ix]), device=device)
                target = torch.tensor(targets[pick], device=device)
                mask = torch.tensor(allowed[pick], device=device)
                logits = net(uid, ids, hist, ages).masked_fill(~mask, -1e4)
                loss = F.cross_entropy(logits, target)
                active_optimizer = optimizer if e > 0 else head_optimizer
                if e > 0:
                    p = torch.randint(len(au), (len(ix),), device=device)
                    uu, ii = au[p], ai[p]
                    ni = torch.randint(1, len(initial['items']), (len(ix), 8), device=device)
                    a, b, c = net.users(uu), net.items(ii), net.items(ni)
                    pos = (a*b).sum(1)+net.bias[ii]-np.log(8)
                    neg = (a[:, None]*c).sum(2)+net.bias[ni]-np.log(8)
                    nce = (F.softplus(-pos).mean()+8*F.softplus(neg).mean())/9
                    penalty = F.mse_loss(net.users(uid), net.u0[uid])+F.mse_loss(net.items(ids), net.v0[ids])
                    loss = loss+.3*nce+.5*penalty
                active_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                active_optimizer.step()
                total += float(loss.item())*len(ix)
            if e > 0:
                schedule.step()
            row = dict(epoch=e, loss=total/len(indices), seconds=time.monotonic()-tick,
                       coefficients=net.coefficients.detach().cpu().tolist())
            records.append(row)
            if e in [0, 12, 40]:
                u = net.users.weight.detach().cpu().numpy().copy()
                v = net.items.weight.detach().cpu().numpy().copy()
                u[np.diff(x.indptr) == 0] = 0
                v[np.asarray(x.sum(0)).ravel() == 0] = 0
                u[0] = v[0] = 0
                np.savez_compressed(folder/('e'+str(e)+'.npz'), users=u, items=v, bias=initial['bias'])
            if e % 4 == 0:
                print(json.dumps(dict(stage='novel_choice', cutoff=cutoff, **row)), flush=True)
                write_json(folder/'training.json', records)
        write_json(folder/'result.json', dict(cutoff=cutoff, events=len(indices), max_label_timestamp=int(raw['timestamps'][indices].max()),
                   epochs=40, warmup=8, anchor=.5, historical_nce=.3, seed=4187,
                   rule='Only clicked events whose target is absent from their visible history; all repeat candidate slots excluded.'))
    return dict(np.load(folder/('e'+str(epoch)+'.npz')))


def extra_features(data, indices, cutoff, config=None):
    from arena_coclick import describe
    pieces, names = [], []
    for epoch in [12, 40]:
        x, cols = describe(model(cutoff, epoch), data, indices)
        pieces.append(x)
        names += [n.replace('coclick_', 'novel'+str(epoch)+'_', 1) for n in cols]
    return np.concatenate(pieces, 1), names
