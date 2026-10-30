"""Read-only experiment: spherical SGNS8 with past-only fitting positives.

This does not export a submission or alter an accepted model. The normalization
and checkpoints are fixed before the two diagnostic validation origins.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import sparse
from sklearn.metrics import roc_auc_score
from arena_ood import ROOT, guard, write_json


class Spherical(nn.Module):
    def __init__(self, initial, known_users, known_items):
        super().__init__()
        u, v = [F.normalize(torch.tensor(initial[k]), dim=1, eps=1e-6) for k in ['users', 'items']]
        u[~known_users] = 0
        v[~known_items] = 0
        u[0] = v[0] = 0
        self.users = nn.Embedding.from_pretrained(u.clone(), freeze=False, padding_idx=0)
        self.items = nn.Embedding.from_pretrained(v.clone(), freeze=False, padding_idx=0)
        self.register_buffer('u0', u)
        self.register_buffer('v0', v)
        self.register_buffer('known_users', torch.tensor(known_users))
        self.register_buffer('known_items', torch.tensor(known_items))
        self.user_bias = nn.Embedding(len(u), 1, padding_idx=0)
        nn.init.zeros_(self.user_bias.weight)
        self.item_bias = nn.Embedding.from_pretrained(torch.tensor(initial['bias'])[:, None].clone(), freeze=False, padding_idx=0)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(6.)))

    def lookup(self, ids, users=False):
        weight = self.users(ids) if users else self.items(ids)
        known = self.known_users[ids] if users else self.known_items[ids]
        return F.normalize(weight, dim=-1, eps=.1)*known[..., None]

    def loss(self, uid, positive, negative):
        u, v, n = self.lookup(uid, True), self.lookup(positive), self.lookup(negative)
        temperature = self.log_temperature.exp().clamp(1., 15.)
        shift = math.log(negative.shape[1])
        bu = self.user_bias(uid)
        pos = temperature*(u*v).sum(1)+self.item_bias(positive).squeeze(1)+bu.squeeze(1)-shift
        neg = temperature*(u[:, None]*n).sum(2)+self.item_bias(negative).squeeze(2)+bu-shift
        nce = (F.softplus(-pos).mean()+negative.shape[1]*F.softplus(neg).mean())/(negative.shape[1]+1)
        penalty = (u-self.u0[uid]).square().mean()+(v-self.v0[positive]).square().mean()+(n-self.v0[negative]).square().mean()
        return nce+.02*penalty

    @torch.no_grad()
    def representation(self):
        uid = torch.arange(len(self.u0), device=self.u0.device)
        iid = torch.arange(len(self.v0), device=self.v0.device)
        return dict(users=self.lookup(uid, True).cpu().numpy(), items=self.lookup(iid).cpu().numpy(),
                    bias=self.item_bias.weight.detach().cpu().numpy().ravel())


def run(day, epochs=32):
    from arena_core import snapshot
    from arena_sgns_features import embedding_model
    from ctr_data import load_cache, temporal_split
    from run import timestamp
    cutoff = '2025-01-'+day
    folder = ROOT/'competition/spherical_probe'/day
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'report.json').exists():
        return
    before = guard()
    initial = embedding_model(cutoff, 8, recipe=dict(epochs=320, lr=.01, lr_decay=False, seed=2026,
                                                  reuse_experiment='sgns8_const_e320_s2026'))
    x = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
    known_u, known_i = np.diff(x.indptr)>0, np.asarray(x.sum(0)).ravel()>0
    known_u[0] = known_i[0] = False
    pu, pi = x.nonzero()
    _, raw = load_cache(ROOT/'data/cache_v1', 'train')
    _, valid = temporal_split(raw, timestamp(cutoff))
    torch.manual_seed(82719)
    torch.set_num_threads(2)
    model = Spherical(initial, known_u, known_i)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.0002)
    pu, pi = torch.tensor(pu, dtype=torch.long), torch.tensor(pi, dtype=torch.long)
    record, started = [], time.monotonic()
    for epoch in range(epochs+1):
        if epoch:
            tick, total = time.monotonic(), 0.
            order = torch.randperm(len(pu))
            for start in range(0, len(order), 8192):
                take = order[start:start+8192]
                uid, iid = pu[take], pi[take]
                negative = torch.randint(1, len(initial['items']), (len(take), 8))
                loss = model.loss(uid, iid, negative)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                with torch.no_grad():
                    for q, p in zip(ema.parameters(), model.parameters()):
                        q.mul_(.99).add_(p, alpha=.01)
                total += loss.item()*len(take)
            scheduler.step()
            print(json.dumps(dict(day=day, epoch=epoch, loss=total/len(pu), seconds=time.monotonic()-tick)), flush=True)
        if epoch in [0, 8, 16, epochs]:
            rep = ema.representation()
            u, v = rep['users'][raw['users'][valid]], rep['items'][raw['items'][valid]]
            score = (u[:, None]*v).sum(2)
            auc = float(roc_auc_score(raw['labels'][valid].ravel(), score.ravel()))
            row = dict(epoch=epoch, cosine_auc=auc, temperature=float(ema.log_temperature.exp()))
            record.append(row)
            np.savez(folder/('factors_e'+str(epoch)+'.npz'), **rep)
            print(json.dumps(dict(stage='spherical_diagnostic', day=day, **row)), flush=True)
    # Other independent campaigns may advance CURRENT_BEST while this read-only
    # diagnostic runs; all pre-existing immutable artifacts must remain intact.
    from arena_ood import sha
    assert all(sha(ROOT/path)==digest for path,digest in before.items() if path!='competition/CURRENT_BEST.json')
    write_json(folder/'report.json', dict(cutoff=cutoff, epochs=epochs, seed=82719, records=record,
               seconds=time.monotonic()-started, source='Past-only binary click matrix; validation labels used for diagnostics only.',
               snapshot_metadata=json.loads((snapshot(cutoff)/'metadata.json').read_text())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', choices=['26', '25'], required=True)
    args = parser.parse_args()
    run(args.day)
