"""Exposure MAP-MF with support-weighted priors and an honest earlier anchor.

Global score calibration uses later fitting events with the anchor factors
frozen. Its scale is then fixed while factors and ID intercepts are updated.
The per-ID prior is divided by its fitting exposure count, so SGD applies each
ID prior once per dataset rather than once per occurrence.
"""
import copy
import fcntl
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.55')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.4')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_ood import ROOT, OUT, guard, sha, write_json

ANCILLARY = ['ub', 'ib', 'u_precision', 'v_precision', 'bu_precision', 'bi_precision', 'head_users', 'head_items']


def anchor(cutoff):
    if cutoff == '2025-01-27':
        return '2025-01-23'
    return (datetime.strptime(cutoff, '%Y-%m-%d')-timedelta(days=3)).strftime('%Y-%m-%d')


def fit_indices(data, cutoff, full=False):
    from run import timestamp
    mask = data['timestamps'] < timestamp(cutoff)
    if not full:
        mask &= data['timestamps'] >= timestamp(anchor(cutoff))
    return np.flatnonzero(mask)


class BayesianMF(nn.Module):
    def __init__(self, initial, ancillary, config):
        super().__init__()
        self.config = config
        u, v = torch.as_tensor(initial['users'], dtype=torch.float32), torch.as_tensor(initial['items'], dtype=torch.float32)
        self.register_buffer('u0', u.clone())
        self.register_buffer('v0', v.clone())
        self.users = nn.Embedding.from_pretrained(u.clone(), freeze=False, padding_idx=0)
        self.items = nn.Embedding.from_pretrained(v.clone(), freeze=not config.get('fine_items', True), padding_idx=0)
        self.user_delta = nn.Embedding(len(u), 1, padding_idx=0)
        self.item_delta = nn.Embedding(len(v), 1, padding_idx=0)
        nn.init.zeros_(self.user_delta.weight)
        nn.init.zeros_(self.item_delta.weight)
        for key in ANCILLARY:
            self.register_buffer(key, torch.as_tensor(ancillary[key], dtype=torch.float32))
        # dot, user/item CTR priors, fixed history similarities, repeats, context.
        self.coefficients = nn.Parameter(torch.tensor([.25, 1., 1., .5, .3, .5, 1.5, .3, .2, .2]+[0.]*6))
        self.outside = nn.Parameter(torch.tensor(2.8))

    def configure(self, fine):
        self.requires_grad_(False)
        if fine:
            self.users.requires_grad_(True)
            self.items.requires_grad_(self.config.get('fine_items', True))
            self.user_delta.requires_grad_(True)
            self.item_delta.requires_grad_(True)
        else:
            self.coefficients.requires_grad_(True)
            self.outside.requires_grad_(True)

    def features(self, batch):
        uid, ids, hist = batch['users'], batch['items'], batch['history']
        age, ctx = batch['age_hours'], batch['context']
        dot = (self.users(uid)[:, None]*self.items(ids)).sum(2)
        # Only the latent dot term receives factor gradients. Fixed historical
        # covariates cannot be re-shaped into a fitting-click shortcut.
        v, h = self.v0[ids], self.v0[hist]
        v = F.normalize(v, dim=2, eps=1e-6)
        h = F.normalize(h, dim=2, eps=1e-6)
        similarity = torch.bmm(v, h.transpose(1, 2))
        visible = hist != 0
        length = visible.sum(1)
        mean = (similarity*visible[:, None]).sum(2)/length.clamp_min(1)[:, None]
        peak = similarity.masked_fill(~visible[:, None], -1).max(2).values
        peak = torch.where(length[:, None] > 0, peak, 0)
        weight = torch.exp(-age/72)*visible
        recent = (similarity*weight[:, None]).sum(2)/weight.sum(1).clamp_min(1e-6)[:, None]
        repeat = (ids[:, :, None] == hist[:, None]) & visible[:, None]
        values = [dot, self.ub[uid, None].expand_as(dot), self.ib[ids], mean, peak, recent,
                  repeat.any(2).float(), repeat.sum(2).float().log1p(),
                  (repeat*torch.exp(-age[:, None]/12)).sum(2), (repeat*torch.exp(-age[:, None]/72)).sum(2)]
        return torch.cat([torch.stack(values, 2), ctx[:, None].expand(-1, 10, -1)], 2)

    def forward(self, batch):
        score = (self.features(batch)*self.coefficients).sum(2)
        score = score+self.user_delta(batch['users'])+self.item_delta(batch['items']).squeeze(2)
        return torch.cat([score, self.outside.expand(len(score), 1)], 1)

    def penalty(self, batch):
        uid, ids = batch['users'], batch['items']
        up = (self.users(uid)-self.u0[uid]).square().sum(1)*self.u_precision[uid]/self.head_users[uid].clamp_min(1)
        vp = (self.items(ids)-self.v0[ids]).square().sum(2)*self.v_precision[ids]/self.head_items[ids].clamp_min(1)
        bu = self.user_delta(uid).squeeze(1).square()*self.bu_precision[uid]/self.head_users[uid].clamp_min(1)
        bi = self.item_delta(ids).squeeze(2).square()*self.bi_precision[ids]/self.head_items[ids].clamp_min(1)
        # Sum candidate priors, rather than averaging ten copies of their counts.
        return .5*self.config.get('anchor_scale', 1.)*(up+bu+vp.sum(1)+bi.sum(1)).mean()


def initialize(raw, indices, cutoff, config, prefix=None):
    from arena_core import snapshot
    from arena_sgns_features import embedding_model
    from scipy import sparse
    prefix = prefix or anchor(cutoff)
    m = embedding_model(prefix, 8, recipe=dict(epochs=320, lr=.01, lr_decay=False, seed=2026,
                       reuse_experiment='sgns8_const_e320_s2026'))
    x = sparse.load_npz(snapshot(prefix)/'interactions.npz')
    stats = dict(np.load(snapshot(prefix)/'stats.npz'))
    nu, ni = x.shape
    u, v = m['users'].copy(), m['items'].copy()
    uc, ic = np.diff(x.indptr), np.asarray(x.sum(0)).ravel()
    u[uc == 0], v[ic == 0] = 0, 0
    u[0], v[0] = 0, 0
    ancillary = dict(ub=stats['ub'], ib=stats['ib'], u_precision=np.maximum(uc, 1)*.5,
                     v_precision=np.maximum(ic, 1)*.5,
                     bu_precision=2+(stats['ui']/10)*.63*.37,
                     bi_precision=4+stats['ii']*float(stats['ctr'])*(1-float(stats['ctr'])),
                     head_users=np.bincount(raw['users'][indices], minlength=nu),
                     head_items=np.bincount(raw['items'][indices].ravel(), minlength=ni))
    for key in ['u_precision', 'v_precision', 'bu_precision', 'bi_precision']:
        ancillary[key][0] = 0
    return BayesianMF(dict(users=u, items=v), ancillary, config)


def load_model(path, device):
    saved = torch.load(Path(path)/'model.pt', map_location='cpu', weights_only=True)
    s = saved['state_dict']
    model = BayesianMF(dict(users=s['u0'], items=s['v0']), {k:s[k] for k in ANCILLARY}, saved['config'])
    model.load_state_dict(s)
    model.configure(True)
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, raw, indices, device):
    from run import get_batch
    model.eval()
    parts = []
    for start in range(0, len(indices), 2048):
        parts.append(model(get_batch(raw, indices[start:start+2048], device, labels=False)).softmax(1)[:, :10].cpu().numpy())
    return np.concatenate(parts)


def train(name, cutoff, fixed_epochs, overrides, final=None):
    from ctr_data import load_cache, temporal_split
    from run import timestamp, get_batch
    from arena_core import evaluate
    from arena_validation import importance_weights, weighted_auc
    config, before = overrides['bayes_mf'], guard()
    out = OUT/'runs'/name/cutoff if final is None else final/'models'/name
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'result.json').exists():
            result = json.loads((out/'result.json').read_text())
            assert result['augmentation_config'] == overrides
            return result
        meta, raw = load_cache(ROOT/'data/cache_v1', 'train')
        calibration_fit = fit_indices(raw, cutoff)
        fit = fit_indices(raw, cutoff, full=config.get('full_posterior', False))
        if final is None:
            _, valid = temporal_split(raw, timestamp(cutoff))
            evaluation = raw
            iw = importance_weights(cutoff, valid)[0]
        else:
            _, evaluation = load_cache(ROOT/'data/cache_v1', 'evaluation')
            valid = np.arange(len(evaluation['users']))
        assert len(fit) and int(raw['timestamps'][fit].max()) < timestamp(cutoff)
        if final is None:
            assert not np.intersect1d(fit, valid).size
        torch.manual_seed(config.get('seed', 7153))
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = initialize(raw, fit, cutoff, config).to(device)
        model.configure(False)
        optimizer = torch.optim.AdamW([model.coefficients, model.outside], lr=.003, weight_decay=.01)
        rng, records = np.random.default_rng(config.get('seed', 7153)), []
        started = time.monotonic()

        def epoch_pass(opt, fine):
            model.train()
            active = fit if fine else calibration_fit
            order, total, ce_total = rng.permutation(active), 0., 0.
            for start in range(0, len(order), 2048):
                chosen = order[start:start+2048]
                batch = get_batch(raw, chosen, device, labels=True)
                y = batch['labels']
                logits = model(batch)
                ce = F.cross_entropy(logits, torch.where(y.sum(1)>0, y.argmax(1), 10))
                loss = ce+model.penalty(batch) if fine else ce
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                total += loss.item()*len(chosen)
                ce_total += ce.item()*len(chosen)
                if fine:
                    with torch.no_grad():
                        for target, p in zip(ema.parameters(), model.parameters()):
                            if p.requires_grad:
                                target.mul_(.99).add_(p, alpha=.01)
            return dict(train_loss=total/len(active), cross_entropy=ce_total/len(active))

        for epoch in range(1, 5):
            tick = time.monotonic()
            row = dict(phase='calibration', epoch=epoch, **epoch_pass(optimizer, False), seconds=time.monotonic()-tick)
            records.append(row)
            print(json.dumps(dict(name=name, cutoff=cutoff, **row)), flush=True)
        model.configure(True)
        ema = copy.deepcopy(model).eval()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.get('lr', .002), weight_decay=0)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18, eta_min=.0002)
        fine_records, best, stale = [], -1., 0
        epochs = 18 if fixed_epochs is None else fixed_epochs

        def save(p=None):
            torch.save(dict(state_dict=ema.state_dict(), config=config, cache_id=meta['cache_id']), out/'model.pt')
            if p is not None:
                np.savez_compressed(out/'validation.npz', labels=evaluation['labels'][valid], event_indices=valid, predictions=p)

        for epoch in range(0, epochs+1):
            tick = time.monotonic()
            row = dict(phase='posterior', epoch=epoch)
            if epoch:
                row.update(epoch_pass(optimizer, True))
                schedule.step()
            p = None
            if final is None:
                p = predict(ema, evaluation, valid, device)
                row.update(evaluate(evaluation['labels'][valid], p), weighted_auc=weighted_auc(evaluation['labels'][valid], p, iw))
            row['seconds'] = time.monotonic()-tick
            fine_records.append(row)
            records.append(row)
            print(json.dumps(dict(name=name, cutoff=cutoff, **row)), flush=True)
            chosen = epoch == epochs if fixed_epochs is not None else row['auc'] > best
            if chosen:
                best, stale = row.get('auc', 0.), 0
                save(p)
            else:
                stale += 1
            write_json(out/'training.json', records)
            if fixed_epochs is None and stale >= 4:
                break
        winner = fine_records[-1] if fixed_epochs is not None else max(fine_records, key=lambda r:r['auc'])
        if final is not None:
            np.save(out/'evaluation_predictions.npy', predict(load_model(out, device), evaluation, valid, device))
        result = dict(name=name, cutoff=cutoff, method='exposure_bayesian_mf', config=config, augmentation_config=overrides,
                      best_epoch=winner['epoch'], **{k:v for k,v in winner.items() if k not in ['epoch', 'seconds', 'phase']},
                      anchor=anchor(cutoff), fitting_events=len(fit), max_label_timestamp=int(raw['timestamps'][fit].max()),
                      fitting_indices_sha256=hashlib.sha256(fit.tobytes()).hexdigest(),
                      calibration_epochs=4, parameters=sum(p.numel() for p in model.parameters()),
                      calibration_events=len(calibration_fit),
                      posterior_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      seconds=time.monotonic()-started)
        assert all(sha(ROOT/f) == h for f,h in before.items() if f != 'competition/CURRENT_BEST.json')
        write_json(out/'result.json', result)
        return result


def fit_final(name, final):
    r = json.loads((OUT/'runs'/name/'2025-01-26/result.json').read_text())
    train(name, '2025-01-27', r['best_epoch'], r['augmentation_config'], final)
    path = final/'models'/name
    write_json(path/'recipe.json', dict(name=name, bayes_mf=True, augmentation_config=r['augmentation_config'],
               fixed_epochs=r['best_epoch'], anchor=anchor('2025-01-27')))
    return path
