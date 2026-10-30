"""Joint real-exposure and historical-click training of compact CTR factors."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.55')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import fcntl
import json
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_core import snapshot
from arena_ood import write_json

RECIPE = dict(version=1, factors=8, epochs=32, seed=1401, embedding_lr=.001,
              head_lr=.003, anchor=.05, historical_loss=.3)


class ExposureFactors(nn.Module):
    def __init__(self, initial, stats):
        super().__init__()
        u, v = torch.tensor(initial['users']), torch.tensor(initial['items'])
        self.users = nn.Embedding.from_pretrained(u.clone(), freeze=False, padding_idx=0)
        self.items = nn.Embedding.from_pretrained(v.clone(), freeze=False, padding_idx=0)
        self.register_buffer('u0', u)
        self.register_buffer('v0', v)
        self.register_buffer('nce_bias', torch.tensor(initial['bias']))
        for key in ['ub', 'ib']:
            self.register_buffer(key, torch.tensor(stats[key]))
        self.register_buffer('known', torch.tensor(stats['pop'] > 0))
        self.weight = nn.Parameter(torch.tensor([.25, 1., 2., .3, 2., .3, .7, .7]+[0.]*6))
        self.outside = nn.Parameter(torch.tensor(2.8))

    def features(self, uid, ids, hist, age, ctx):
        hids = torch.where(self.known[hist], hist, 0)
        u, v, h = self.users(uid), self.items(ids), self.items(hids)
        dot = (v*u[:, None]).sum(2)
        cos = (F.normalize(v, dim=2)*F.normalize(u, dim=1)[:, None]).sum(2)
        hc = torch.bmm(F.normalize(v, dim=2), F.normalize(h, dim=2).transpose(1, 2))
        mask = hids != 0
        mean = (hc*mask[:, None]).sum(2)/mask.sum(1).clamp_min(1)[:, None]
        peak = hc.masked_fill(~mask[:, None], -1).max(2).values
        peak = torch.where(mask.any(1)[:, None], peak, 0)
        w = torch.exp(-age/72)*mask
        recent = (hc*w[:, None]).sum(2)/w.sum(1).clamp_min(1e-6)[:, None]
        repeat = ((ids[:, :, None] == hist[:, None]) & (hist[:, None] != 0)).sum(2).float().log1p()
        return torch.cat([torch.stack([dot, cos, mean, peak, recent, repeat,
                          self.ub[uid, None].expand_as(dot), self.ib[ids]], 2), ctx[:, None].expand(-1, 10, -1)], 2)

    def forward(self, uid, ids, hist, age, ctx):
        score = (self.features(uid, ids, hist, age, ctx)*self.weight).sum(2)
        return torch.cat([score, self.outside.expand(len(uid), 1)], 1)


def model(cutoff):
    from ctr_data import load_cache
    from run import timestamp
    from arena_sgns_features import embedding_model
    from scipy import sparse
    folder = snapshot(cutoff)/'exposure_joint_v1_d8_e32'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            return dict(np.load(folder/'factors.npz'))
        initial = embedding_model(cutoff, 8, recipe={'epochs': 320, 'lr': .01, 'lr_decay': False,
                                  'seed': 2026, 'reuse_experiment': 'sgns8_const_e320_s2026'})
        stats = dict(np.load(snapshot(cutoff)/'stats.npz'))
        _, data = load_cache('data/cache_v1', 'train')
        fit = np.flatnonzero(data['timestamps'] < timestamp(cutoff))
        torch.manual_seed(RECIPE['seed'])
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        network = ExposureFactors(initial, stats).to(device)
        optimizer = torch.optim.AdamW([{'params': [network.users.weight, network.items.weight], 'lr': RECIPE['embedding_lr'], 'weight_decay': 0},
                    {'params': [network.weight, network.outside], 'lr': RECIPE['head_lr'], 'weight_decay': .001}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RECIPE['epochs'], eta_min=.0001)
        x = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
        pu, pi = x.nonzero()
        pu, pi = torch.tensor(pu, device=device), torch.tensor(pi, device=device)
        rng = np.random.default_rng(RECIPE['seed'])
        records = []
        for epoch in range(1, RECIPE['epochs']+1):
            start = time.monotonic()
            total = 0.
            for offset in range(0, len(fit), 2048):
                if offset == 0:
                    order = rng.permutation(fit)
                ix = order[offset:offset+2048]
                uid = torch.tensor(np.array(data['users'][ix]), device=device)
                ids = torch.tensor(np.array(data['items'][ix]), device=device)
                hist = torch.tensor(np.array(data['history'][ix]), device=device)
                age = torch.tensor(np.array(data['age_hours'][ix]), device=device)
                ctx = torch.tensor(np.array(data['context'][ix]), device=device)
                y = torch.tensor(np.array(data['labels'][ix]), device=device)
                z = network(uid, ids, hist, age, ctx)
                ce = F.cross_entropy(z, torch.where(y.sum(1) > 0, y.argmax(1), 10))
                pick = torch.randint(len(pu), (len(ix),), device=device)
                au, ai = pu[pick], pi[pick]
                ni = torch.randint(1, len(initial['items']), (len(ix), 8), device=device)
                u, v, n = network.users(au), network.items(ai), network.items(ni)
                pz = (u*v).sum(1)+network.nce_bias[ai]-math.log(8)
                nz = (u[:, None]*n).sum(2)+network.nce_bias[ni]-math.log(8)
                nce = (F.softplus(-pz).mean()+8*F.softplus(nz).mean())/9
                penalty = (network.users(uid)-network.u0[uid]).square().mean()+(network.items(ids)-network.v0[ids]).square().mean()
                loss = ce+RECIPE['historical_loss']*nce+RECIPE['anchor']*penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                total += loss.item()*len(ix)
            scheduler.step()
            row = dict(epoch=epoch, loss=total/len(fit), seconds=time.monotonic()-start)
            records.append(row)
            print(json.dumps(dict(stage='exposure_representation', cutoff=cutoff, **row)), flush=True)
        u, v = network.users.weight.detach().cpu().numpy(), network.items.weight.detach().cpu().numpy()
        # Unobserved IDs cannot acquire information through their random initial vectors.
        u[np.diff(x.indptr) == 0] = 0
        v[stats['pop'] == 0] = 0
        u[0], v[0] = 0, 0
        np.savez(folder/'factors.npz', users=u, items=v, bias=initial['bias'],
                 weight=network.weight.detach().cpu().numpy(), outside=network.outside.detach().cpu().numpy())
        torch.save(network.state_dict(), folder/'training_model.pt')
        write_json(folder/'result.json', dict(cutoff=cutoff, recipe=RECIPE, events=len(fit),
                   max_fitting_timestamp=int(data['timestamps'][fit].max()), epochs=records))
    return dict(np.load(folder/'factors.npz'))


def extra_features(data, indices, cutoff, config=None):
    # Use the same scalar, rotation-invariant descriptors as the co-click view.
    # The profile here is the learned exposure-aware user vector.
    from arena_coclick import describe
    rep = model(cutoff)
    x, names = describe(rep, data, indices)
    return x, [name.replace('coclick_', 'exposure_', 1) for name in names]


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cutoff', default='2025-01-24')
    a = p.parse_args()
    model(a.cutoff)
