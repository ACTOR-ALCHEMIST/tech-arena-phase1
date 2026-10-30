"""Exposure-supervised multi-view MF with online history and user dropout.

Unlike a frozen feature-window head, this scorer trains directly on the full
pre-cutoff exposure set and can fine tune the task-specific embedding tables.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.55')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.4')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_ood import ROOT, write_json, guard, sha

OUT = ROOT/'competition/robust_mf'
RECIPES = {
    'frozen_online': {'fine_tune': False, 'whiten_view': True, 'history_event_probability': .20, 'history_delete_probability': .15, 'user_dropout': .05},
    'tuned_online': {'fine_tune': True, 'whiten_view': True, 'embedding_lr': .00005, 'anchor': .10,
                     'history_event_probability': .20, 'history_delete_probability': .15, 'user_dropout': .05},
    'causal_frozen': {'fine_tune': False, 'whiten_view': True, 'causal_prefix': True, 'lr': .0007, 'ema': .99,
                      'history_event_probability': .20, 'history_delete_probability': .15, 'user_dropout': .05},
    'causal_tuned': {'fine_tune': True, 'whiten_view': True, 'causal_prefix': True, 'lr': .0007, 'ema': .99,
                     'embedding_lr': .0001, 'anchor': .10, 'embedding_warmup_epochs': 2,
                     'history_event_probability': .20, 'history_delete_probability': .15, 'user_dropout': .05},
}


class RobustMF(nn.Module):
    def __init__(self, pretrained, stats, config):
        super().__init__()
        self.config = config
        self.sizes = pretrained['sizes']
        uf, vf = torch.as_tensor(pretrained['users'], dtype=torch.float32), torch.as_tensor(pretrained['items'], dtype=torch.float32)
        self.users = nn.Embedding.from_pretrained(uf.clone(), freeze=not config['fine_tune'], padding_idx=0)
        self.items = nn.Embedding.from_pretrained(vf.clone(), freeze=not config['fine_tune'], padding_idx=0)
        self.register_buffer('user_initial', uf.clone())
        self.register_buffer('item_initial', vf.clone())
        self.register_buffer('known_users', torch.as_tensor(stats['known_users'], dtype=torch.bool))
        self.register_buffer('known_items', torch.as_tensor(stats['pop'] > 0, dtype=torch.bool))
        for key in ['ub', 'ib', 'ui', 'ii']:
            self.register_buffer(key, torch.as_tensor(stats[key], dtype=torch.float32))
        self.register_buffer('global_prior', torch.tensor(float(np.log(stats['ctr']/(1-stats['ctr']))), dtype=torch.float32))
        from arena_neural_v2 import LargeSlateModel
        width = 17*len(self.sizes)+15
        self.head = LargeSlateModel(width, np.zeros(width, np.float32), np.ones(width, np.float32),
                    dict(loss='event_ce', architecture='v2_cross', hidden=[256, 128, 64], context_width=32,
                         cross_layers=2, dropout=config.get('dropout', .10)))

    def raw_features(self, batch, augment=True):
        uid = torch.where(self.known_users[batch['users']], batch['users'], 0)
        original_history = batch['history']
        visible = original_history != 0
        if self.training and augment:
            uid = torch.where(torch.rand(len(uid), device=uid.device) < self.config.get('user_dropout', .05), 0, uid)
            active = torch.rand(len(uid), device=uid.device) < self.config.get('history_event_probability', .20)
            deleted = torch.rand(visible.shape, device=uid.device) < self.config.get('history_delete_probability', .15)
            visible = visible & ~(active[:, None] & deleted)
        history = torch.where(visible & self.known_items[original_history], original_history, 0)
        ids = torch.where(self.known_items[batch['items']], batch['items'], 0)
        age = batch['age_hours']
        count = visible.sum(1)
        rank = visible.cumsum(1)
        last1 = visible & (rank > (count-1)[:, None])
        last5 = visible & (rank > (count-5)[:, None])
        latest = torch.where(visible, age, torch.inf).min(1).values
        latest = torch.where(count > 0, latest, 0)
        oldest = torch.where(visible, age, 0).max(1).values
        ctx = batch['context'].clone()
        ctx[:, 3] = count/20
        ctx[:, 4] = latest.log1p()/np.log1p(720)
        ctx[:, 5] = (oldest-latest).clamp_min(0).log1p()/np.log1p(720)
        past = batch['timestamps'][:, None]//60-torch.round(age*60).long()
        bucket = batch['timestamps']%86400//21600
        same_bucket = ((past%1440)//360 == bucket[:, None]) & visible
        same_weekend = ((((past//1440+3)%7) >= 5) == (ctx[:, 2] > 0)[:, None]) & visible
        u, v, h = self.users(uid), self.items(ids), self.items(history)
        parts = []
        offset = 0
        for size in self.sizes:
            a, b, c = u[:, offset:offset+size], v[:, :, offset:offset+size], h[:, :, offset:offset+size]
            offset += size
            un = a.norm(dim=1).clamp_min(1e-6)
            vn = b.norm(dim=2).clamp_min(1e-6)
            hn = c.norm(dim=2).clamp_min(1e-6)
            dot = (a[:, None, :]*b).sum(2)
            cosine = dot/(un[:, None]*vn)
            hc = torch.bmm(b, c.transpose(1, 2))/(vn[:, :, None]*hn[:, None, :])
            hc = hc*visible[:, None, :]

            def weighted(weights):
                return (hc*weights[:, None, :]).sum(2)/weights.sum(1).clamp_min(1e-6)[:, None]

            mean = weighted(visible)
            variance = ((hc.square()*visible[:, None, :]).sum(2)/count.clamp_min(1)[:, None]-mean.square()).clamp_min(0)
            pooled = (c/hn[:, :, None]*visible[:, :, None]).sum(1)/count.clamp_min(1)[:, None]
            coherence = pooled.norm(dim=1)
            pooled_cos = (b*pooled[:, None, :]).sum(2)/(vn*coherence.clamp_min(1e-6)[:, None])
            values = [dot/size, cosine, un[:, None].expand_as(dot)/np.sqrt(size), vn/np.sqrt(size),
                      mean, hc.max(2).values, weighted(torch.exp(-age/72)*visible), weighted(torch.exp(-age/24)*visible),
                      weighted(last1), weighted(last5), (dot-dot.mean(1, keepdim=True))/size, cosine-cosine.mean(1, keepdim=True),
                      (variance+1e-8).sqrt(), pooled_cos, coherence[:, None].expand_as(dot), weighted(same_bucket), weighted(same_weekend)]
            parts.append(torch.stack(values, dim=2))
        repeat = (batch['items'][:, :, None] == original_history[:, None, :]) & visible[:, None, :]
        ub, ib = self.ub[uid], self.ib[batch['items']]
        extras = [self.global_prior+ub[:, None]+ib, ub[:, None].expand_as(ib), ib,
                  self.ui[uid].log1p()[:, None].expand_as(ib), self.ii[batch['items']].log1p(),
                  repeat.any(2).float(), repeat.sum(2).float().log1p(),
                  (repeat*torch.exp(-age[:, None, :]/12)).sum(2), (repeat*torch.exp(-age[:, None, :]/72)).sum(2)]
        return torch.cat(parts+[ctx[:, None, :].expand(-1, 10, -1), torch.stack(extras, dim=2)], dim=2)

    def forward(self, batch):
        return self.head(self.raw_features(batch))

    def probabilities(self, batch):
        return self(batch).softmax(1)[:, :10]

    def penalty(self, batch):
        if not self.config['fine_tune']:
            return torch.zeros((), device=batch['users'].device)
        uid = batch['users']
        iid = torch.cat([batch['items'], batch['history']], dim=1)
        return self.config.get('anchor', .10)*(F.mse_loss(self.users(uid), self.user_initial[uid])+
                                              F.mse_loss(self.items(iid), self.item_initial[iid]))


def batch(data, indices, device, labels=False):
    from run import get_batch
    result = get_batch(data, indices, device, labels=labels)
    result['timestamps'] = torch.as_tensor(np.array(data['timestamps'][indices]), dtype=torch.long, device=device)
    return result


@torch.inference_mode()
def predict(model, data, indices, device):
    model.eval()
    values = []
    for start in range(0, len(indices), 1024):
        values.append(model.probabilities(batch(data, indices[start:start+1024], device)).cpu().numpy())
    return np.concatenate(values)


def pretraining_cutoff(cutoff, config):
    if not config.get('causal_prefix'):
        return cutoff
    # Final refit retains the primary Jan24 anchor and adds Jan26 exposures.
    if cutoff == '2025-01-27':
        return '2025-01-24'
    return (datetime.strptime(cutoff, '%Y-%m-%d')-timedelta(days=2)).strftime('%Y-%m-%d')


def initialize(cutoff, config):
    from arena_ood import source
    from arena_sgns_features import embedding_model
    from arena_core import snapshot
    cutoff = pretraining_cutoff(cutoff, config)
    fc = source('2025-01-26')['feature_config']
    specs = [(f, fc.get('sgns_recipes', {}).get(str(f)), fc['sgns_epochs'][str(f)]) for f in fc['sgns_factors']]
    specs += [(x['factors'], x['recipe'], None) for x in fc['sgns_extra']]
    models = [embedding_model(cutoff, f, recipe=r, epochs=e) for f, r, e in specs]
    sizes = [s[0] for s in specs]
    if config.get('whiten_view'):
        stats = np.load(snapshot(cutoff)/'stats.npz')
        common = stats['pop'] > 0
        # Item-space whitening recovers an isotropic history metric; the dual
        # user transform preserves within-slate dot products. All fit is past-only.
        item = models[0]['items'].astype(np.float64)
        center = item[common].mean(0)
        centered = item-center
        covariance = centered[common].T@centered[common]/common.sum()
        values, vectors = np.linalg.eigh(covariance)
        regularized = values+.1*float(np.trace(covariance))/len(values)
        transform = (vectors*(1/np.sqrt(regularized.clip(1e-8))))@vectors.T
        whitened = centered@transform
        dual_users = models[0]['users']@np.linalg.inv(transform).T
        whitened[~common] = 0
        whitened[0] = 0
        dual_users[0] = 0
        models.append({'users': dual_users.astype(np.float32), 'items': whitened.astype(np.float32)})
        sizes.append(specs[0][0])
    pretrained = dict(users=np.concatenate([m['users'] for m in models], axis=1),
                      items=np.concatenate([m['items'] for m in models], axis=1), sizes=sizes)
    return RobustMF(pretrained, dict(np.load(snapshot(cutoff)/'stats.npz')), config)


def load_model(path, device):
    saved = torch.load(Path(path)/'model.pt', map_location='cpu', weights_only=True)
    s = saved['state_dict']
    pretrained = {'users': s['user_initial'].numpy(), 'items': s['item_initial'].numpy(), 'sizes': saved['sizes']}
    stats = {k: s[k].numpy() for k in ['known_users', 'ub', 'ib', 'ui', 'ii']}
    stats.update(pop=s['known_items'].numpy().astype(np.float32), ctr=float(s['global_prior'].sigmoid()))
    model = RobustMF(pretrained, stats, saved['config'])
    model.load_state_dict(s)
    model.to(device)
    model.eval()
    return model


def refresh_model(model, cutoff, device):
    """Keep scalar scoring weights; refresh only the past-only snapshot inputs.

    Scalar dot/cosine features transfer across embedding coordinate systems.
    For a tuned scorer, rotate the small learned deltas into the refreshed frame.
    """
    config = dict(model.config, causal_prefix=False)
    fresh = initialize(cutoff, config).to(device)
    fresh.head.load_state_dict(model.head.state_dict())
    if model.config['fine_tune']:
        old_item = model.item_initial.detach().cpu().numpy()
        new_item = fresh.item_initial.detach().cpu().numpy()
        delta_u = (model.users.weight-model.user_initial).detach().cpu().numpy()
        delta_i = (model.items.weight-model.item_initial).detach().cpu().numpy()
        common = model.known_items.detach().cpu().numpy() & fresh.known_items.detach().cpu().numpy()
        moved_u, moved_i, offset = [], [], 0
        for size in model.sizes:
            a, b = old_item[common, offset:offset+size], new_item[common, offset:offset+size]
            left, _, right = np.linalg.svd(a.astype(np.float64).T@b.astype(np.float64))
            rotation = (left@right).astype(np.float32)
            moved_u.append(delta_u[:, offset:offset+size]@rotation)
            moved_i.append(delta_i[:, offset:offset+size]@rotation)
            offset += size
        with torch.no_grad():
            fresh.users.weight.add_(torch.tensor(np.concatenate(moved_u, 1), device=device))
            fresh.items.weight.add_(torch.tensor(np.concatenate(moved_i, 1), device=device))
    fresh.eval()
    return fresh


def train(name, cutoff='2025-01-26', fixed_epochs=None, final=False):
    from ctr_data import load_cache, temporal_split
    from run import timestamp
    from arena_core import evaluate
    from arena_validation import importance_weights, weighted_auc
    config = {**dict(epochs=12, patience=3, seed=31415, lr=.001, dropout=.1, ema=.995), **RECIPES[name]}
    before = guard()
    path = OUT/'runs'/name/cutoff
    path.mkdir(parents=True, exist_ok=True)
    with (path/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (path/'result.json').exists():
            result = json.loads((path/'result.json').read_text())
            assert result['config'] == config, 'Use a new run name for changed parameters'
            return result
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        meta, data = load_cache(ROOT/'data/cache_v1', 'train')
        if final:
            fit = np.arange(len(data['users']))
            _, evaluation = load_cache(ROOT/'data/cache_v1', 'evaluation')
            valid = np.arange(len(evaluation['users']))
        else:
            fit, valid = temporal_split(data, timestamp(cutoff))
            evaluation = data
        initial_cutoff = pretraining_cutoff(cutoff, config)
        if config.get('causal_prefix'):
            fit = fit[data['timestamps'][fit] >= timestamp(initial_cutoff)]
            assert len(fit) and int(data['timestamps'][fit].min()) >= timestamp(initial_cutoff)
        torch.manual_seed(config['seed'])
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        write_json(path/'config.json', dict(config=config, cache_id=meta['cache_id'], pretraining_cutoff=initial_cutoff,
                   fitting_events=len(fit), min_fitting_timestamp=int(data['timestamps'][fit].min()),
                   max_fitting_timestamp=int(data['timestamps'][fit].max()), validation_cutoff=cutoff,
                   fitting_indices_sha256=hashlib.sha256(fit.tobytes()).hexdigest()))
        model = initialize(cutoff, config).to(device)
        # Fit the normalizer on a fixed clean fitting sample, before all updates.
        chosen = np.sort(np.random.default_rng(9001).choice(fit, min(30000, len(fit)), replace=False))
        features = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(chosen), 1024):
                features.append(model.raw_features(batch(data, chosen[start:start+1024], device), augment=False).cpu().numpy())
        x = np.concatenate(features).reshape(-1, model.head.mean.numel())
        from arena_tabular import normalizer
        mean, scale = normalizer(x, 'zero')
        model.head.mean.copy_(torch.tensor(mean, device=device))
        model.head.scale.copy_(torch.tensor(scale, device=device))
        del x, features
        ema = copy.deepcopy(model).eval()
        embeddings = [p for p in list(model.users.parameters())+list(model.items.parameters()) if p.requires_grad]
        optimizer = torch.optim.AdamW([{'params': model.head.parameters(), 'lr': config['lr'], 'weight_decay': .01},
                                       {'params': embeddings, 'lr': config.get('embedding_lr', .00005), 'weight_decay': 0}])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=.00002)
        rng = np.random.default_rng(config['seed'])
        iw = importance_weights(cutoff, valid)[0] if not final else None
        records, best, stale = [], -1., 0
        started = time.monotonic()
        for epoch in range(1, (fixed_epochs or config['epochs'])+1):
            tick = time.monotonic()
            model.train()
            if config['fine_tune']:
                enable_embeddings = epoch > config.get('embedding_warmup_epochs', 0)
                model.users.requires_grad_(enable_embeddings)
                model.items.requires_grad_(enable_embeddings)
            order = rng.permutation(fit)
            total = 0.
            for offset in range(0, len(order), 2048):
                selected = order[offset:offset+2048]
                inputs = batch(data, selected, device, labels=True)
                labels = inputs['labels']
                logits = model(inputs)
                loss = F.cross_entropy(logits, torch.where(labels.sum(1)>0, labels.argmax(1), 10))+model.penalty(inputs)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                total += float(loss.item())*len(selected)
                with torch.no_grad():
                    for target, parameter in zip(ema.parameters(), model.parameters()):
                        if parameter.requires_grad:
                            target.mul_(config['ema']).add_(parameter, alpha=1-config['ema'])
            scheduler.step()
            row = dict(epoch=epoch, train_loss=total/len(fit), seconds=time.monotonic()-tick)
            if not final:
                p = predict(ema, evaluation, valid, device)
                row.update(evaluate(data['labels'][valid], p), weighted_auc=weighted_auc(data['labels'][valid], p, iw))
            records.append(row)
            print(json.dumps(dict(name=name, cutoff=cutoff, **row)), flush=True)
            save = epoch==fixed_epochs if fixed_epochs else row['auc']>best
            if save:
                best = row.get('auc', 0.)
                stale = 0
                torch.save(dict(state_dict=ema.state_dict(), config=config, sizes=model.sizes, cache_id=meta['cache_id']), path/'model.pt')
                if not final:
                    np.savez_compressed(path/'validation.npz', labels=data['labels'][valid], event_indices=valid, predictions=p)
            else:
                stale += 1
            write_json(path/'training.json', records)
            if fixed_epochs is None and stale >= config['patience']:
                break
        best_row = records[-1] if fixed_epochs else max(records, key=lambda r: r['auc'])
        if final:
            saved = torch.load(path/'model.pt', map_location='cpu', weights_only=True)
            ema.load_state_dict(saved['state_dict'])
            p = predict(ema, evaluation, valid, device)
            np.save(path/'evaluation_predictions.npy', p)
        result = dict(name=name, cutoff=cutoff, method='robust_multiview_mf', config=config,
                      pretraining_cutoff=initial_cutoff,
                      training_events=len(fit), best_epoch=best_row['epoch'], **{k:v for k,v in best_row.items() if k not in ['epoch','seconds']},
                      parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      seconds=time.monotonic()-started, incumbent_unchanged=guard()==before)
        assert result['incumbent_unchanged']
        write_json(path/'result.json', result)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, choices=list(RECIPES))
    p.add_argument('--cutoff', default='2025-01-26')
    p.add_argument('--fixed-epochs', type=int)
    p.add_argument('--final', action='store_true')
    a = p.parse_args()
    if a.final and (a.cutoff != '2025-01-27' or not a.fixed_epochs):
        p.error('Final refit requires cutoff Jan27 and locked positive epoch count')
    train(a.name, a.cutoff, a.fixed_epochs, a.final)
