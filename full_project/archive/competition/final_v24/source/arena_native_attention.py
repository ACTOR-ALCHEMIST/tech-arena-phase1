"""Past-only, same-frame history attention on a frozen strong numeric scorer.

The representation anchor precedes all supervised head events. Embeddings are
stored inside the checkpoint, so inference never changes the learned frame.
"""
import copy
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.55')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.4')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from arena_ood import ROOT, OUT, guard, sha, write_json


class HistoryAttention(nn.Module):
    def __init__(self, base, pretrained, mean, scale, config):
        super().__init__()
        self.base, self.config, self.sizes = base, config, pretrained['sizes']
        self.base.requires_grad_(config.get('joint_base', False))
        u = torch.as_tensor(pretrained['users'], dtype=torch.float32)
        v = torch.as_tensor(pretrained['items'], dtype=torch.float32)
        self.users = nn.Embedding.from_pretrained(u.clone(), freeze=not config.get('fine_tune'), padding_idx=0)
        self.items = nn.Embedding.from_pretrained(v.clone(), freeze=not config.get('fine_tune'), padding_idx=0)
        self.register_buffer('user_initial', u.clone())
        self.register_buffer('item_initial', v.clone())
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        d, width = u.shape[1], config.get('width', 128)
        dropout = config.get('dropout', .1)
        from arena_neural_v2 import mlp
        self.numeric = mlp([len(mean), 96, 64], dropout)
        self.query = nn.Sequential(nn.Linear(3*d+64+6, width), nn.LayerNorm(width), nn.SiLU())
        self.history_input = nn.Linear(d+5, width)
        self.position = nn.Embedding(21, width, padding_idx=0)
        self.history_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, 4, width*2, dropout, batch_first=True, norm_first=True)
            for _ in range(config.get('history_layers', 1))])
        self.attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.slate = (nn.TransformerEncoderLayer(width, 4, width*2, dropout, batch_first=True, norm_first=True)
                      if config.get('slate', True) else nn.Identity())
        self.score = mlp([width*3, 128, 64, 1], dropout)
        self.outside = mlp([width*4, 64, 1], dropout)
        for last in [self.score[-1], self.outside[-1]]:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def train(self, mode=True):
        super().train(mode)
        if not self.config.get('joint_base', False):
            self.base.eval()
        return self

    def normalize_views(self, value):
        # Smooth at zero: eps-only unit normalization makes cold-ID gradients huge.
        pieces, start = [], 0
        for size in self.sizes:
            a = value[..., start:start+size]
            pieces.append(a/torch.sqrt(1+a.square().sum(-1, keepdim=True)))
            start += size
        return torch.cat(pieces, -1)

    def forward(self, batch, x):
        base_logits = self.base(x[:, :, :self.base.mean.numel()])
        history, uid = batch['history'], batch['users']
        visible = history != 0
        if self.training:
            visible = visible & (torch.rand(visible.shape, device=x.device) >= self.config.get('history_dropout', .1))
            uid = torch.where(torch.rand(uid.shape, device=x.device) < self.config.get('user_dropout', .05), 0, uid)
        u = self.normalize_views(self.users(uid))
        v = self.normalize_views(self.items(batch['items']))
        h = self.normalize_views(self.items(torch.where(visible, history, 0)))
        age = batch['age_hours'].clamp(0, 24*365)
        ages = torch.stack([age.log1p()/np.log1p(720), torch.exp(-age/72),
                            torch.sin(age*(2*np.pi/24)), torch.cos(age*(2*np.pi/24)), visible.float()], -1)
        count, rank = visible.sum(1), visible.cumsum(1)
        relative = torch.where(visible, (count[:, None]-rank+1).clamp(0, 20), 0)
        tokens = self.history_input(torch.cat([h, ages], -1))+self.position(relative)
        tokens = tokens*visible[:, :, None]
        safe = visible.clone()
        safe[:, 0] |= count == 0
        for layer in self.history_layers:
            tokens = layer(tokens, src_key_padding_mask=~safe)
        numeric = self.numeric(((x-self.mean)/self.scale).clamp(-6, 6))
        ue = u[:, None, :].expand(-1, 10, -1)
        ctx = batch['context'][:, None, :].expand(-1, 10, -1)
        q = self.query(torch.cat([ue, v, ue*v, numeric, ctx], -1))
        read, _ = self.attention(q, tokens, tokens, key_padding_mask=~safe, need_weights=False)
        read = read*(count > 0)[:, None, None]
        slate = self.slate(q+read)
        residual = self.score(torch.cat([q, read, slate], -1)).squeeze(-1)
        aggregate = torch.cat([q.mean(1), q.max(1).values, read.mean(1), slate.mean(1)], -1)
        outside = self.outside(aggregate)
        return base_logits+self.config.get('residual_scale', .5)*torch.cat([residual, outside], 1)

    def penalty(self, batch):
        if not self.config.get('fine_tune'):
            return torch.zeros((), device=batch['users'].device)
        uid, iid = batch['users'], torch.cat([batch['items'], batch['history']], 1)
        return self.config.get('anchor', .1)*(F.mse_loss(self.users(uid), self.user_initial[uid])+
                                            F.mse_loss(self.items(iid), self.item_initial[iid]))


def initial(cutoff, config):
    from arena_core import snapshot
    from arena_sgns_features import embedding_model
    from arena_ood import source
    fc = source('2025-01-26')['feature_config']
    specs = [(f, fc.get('sgns_recipes', {}).get(str(f)), fc['sgns_epochs'][str(f)]) for f in fc['sgns_factors']]
    specs += [(x['factors'], x['recipe'], None) for x in fc['sgns_extra']]
    stats = np.load(snapshot(cutoff)/'stats.npz')
    users, items = [], []
    for f, recipe, epochs in specs:
        m = embedding_model(cutoff, f, epochs=epochs, recipe=recipe)
        u, v = m['users'].copy(), m['items'].copy()
        u[~stats['known_users']] = 0
        v[stats['pop'] <= 0] = 0
        u[0] = v[0] = 0
        users.append(u)
        items.append(v)
    return dict(users=np.concatenate(users, 1), items=np.concatenate(items, 1), sizes=[x[0] for x in specs])


def base_model(path):
    from arena_push_model import build_model
    s = torch.load(Path(path)/'model.pt', map_location='cpu', weights_only=True)
    model = build_model(s['width'], s['state_dict']['mean'].numpy(), s['state_dict']['scale'].numpy(), s['config'])
    model.load_state_dict(s['state_dict'])
    return model, s['config']


def load_model(path, device):
    from arena_push_model import build_model
    saved = torch.load(Path(path)/'model.pt', map_location='cpu', weights_only=True)
    s, cfg = saved['state_dict'], saved['config']
    base = build_model(s['base.mean'].numel(), s['base.mean'].numpy(), s['base.scale'].numpy(), saved['base_config'])
    pre = dict(users=s['user_initial'].numpy(), items=s['item_initial'].numpy(), sizes=saved['sizes'])
    model = HistoryAttention(base, pre, s['mean'].numpy(), s['scale'].numpy(), cfg)
    model.load_state_dict(s)
    return model.to(device).eval()


def batch(data, indices, device):
    from run import get_batch
    return get_batch(data, indices, device, labels=False)


@torch.inference_mode()
def predict(model, raw, indices, x, device):
    model.eval()
    predictions = []
    for start in range(0, len(indices), 512):
        ix = indices[start:start+512]
        values = np.array(x[start*10:(start+len(ix))*10]).reshape(-1, 10, x.shape[1])
        z = model(batch(raw, ix, device), torch.tensor(values, device=device))
        predictions.append(z.softmax(1)[:, :10].cpu().numpy())
    return np.concatenate(predictions)


def train_loop(name, cutoff, config, raw, fit, evaluation, valid, x, v, base_path, anchor_cutoff,
               out, fixed_epochs=None, final=False):
    from arena_core import evaluate
    from arena_push_model import normalizer
    from arena_validation import importance_weights, weighted_auc
    torch.manual_seed(config.get('seed', 31415))
    torch.set_num_threads(2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    base, base_cfg = base_model(base_path)
    mean, scale = normalizer(x, {'normalizer': 'relative_range'})
    model = HistoryAttention(base, initial(anchor_cutoff, config), mean, scale, config).to(device)
    ema = copy.deepcopy(model).eval()
    heads = [p for k, p in model.named_parameters() if not k.startswith(('base.', 'users.', 'items.'))]
    groups = [dict(params=heads, lr=config.get('lr', .0007), weight_decay=.01)]
    if config.get('joint_base'):
        groups.append(dict(params=model.base.parameters(), lr=config.get('base_lr', .00005), weight_decay=.01))
    if config.get('fine_tune'):
        groups.append(dict(params=list(model.users.parameters())+list(model.items.parameters()),
                           lr=config.get('embedding_lr', .00005), weight_decay=0))
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('epochs', 15), eta_min=.00001)
    rng = np.random.default_rng(config.get('seed', 31415))
    y = raw['labels'][fit]
    assert len(fit)*10 == len(x)
    iw = importance_weights(cutoff, valid)[0] if not final else None
    started, records, best, stale = time.monotonic(), [], -1., 0
    epochs = config.get('epochs', 15) if fixed_epochs is None else fixed_epochs

    def save(epoch, prediction):
        torch.save(dict(state_dict=ema.state_dict(), config=config, base_config=base_cfg, sizes=model.sizes), out/'model.pt')
        if prediction is not None:
            np.savez_compressed(out/'validation.npz', event_indices=valid, labels=evaluation['labels'][valid], predictions=prediction)

    if not final:
        p = predict(ema, evaluation, valid, v, device)
        row = dict(epoch=0, train_loss=None, **evaluate(evaluation['labels'][valid], p),
                   weighted_auc=weighted_auc(evaluation['labels'][valid], p, iw))
        records.append(row)
        best = row['auc']
        if fixed_epochs is None or fixed_epochs == 0:
            save(0, p)
        print(json.dumps(dict(name=name, cutoff=cutoff, **row)), flush=True)
    elif epochs == 0:
        save(0, None)
        records.append(dict(epoch=0, train_loss=None))
    for epoch in range(1, epochs+1):
        tick = time.monotonic()
        model.train()
        order, total = rng.permutation(len(fit)), 0.
        for start in range(0, len(order), 512):
            chosen = order[start:start+512]
            rows = (chosen[:, None]*10+np.arange(10)).ravel()
            bx = torch.tensor(np.array(x[rows]).reshape(-1, 10, x.shape[1]), device=device)
            inputs = batch(raw, fit[chosen], device)
            by = torch.tensor(y[chosen], device=device)
            logits = model(inputs, bx)
            loss = F.cross_entropy(logits, torch.where(by.sum(1)>0, by.argmax(1), 10))+model.penalty(inputs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.item())*len(chosen)
            with torch.no_grad():
                for target, param in zip(ema.parameters(), model.parameters()):
                    if param.requires_grad:
                        target.mul_(config.get('ema', .99)).add_(param, alpha=1-config.get('ema', .99))
        scheduler.step()
        row = dict(epoch=epoch, train_loss=total/len(fit), seconds=time.monotonic()-tick)
        p = None
        if not final:
            p = predict(ema, evaluation, valid, v, device)
            row.update(evaluate(evaluation['labels'][valid], p), weighted_auc=weighted_auc(evaluation['labels'][valid], p, iw))
        records.append(row)
        print(json.dumps(dict(name=name, cutoff=cutoff, **row)), flush=True)
        improved = epoch == epochs if fixed_epochs is not None else row['auc'] > best
        if improved:
            best, stale = row.get('auc', 0.), 0
            save(epoch, p)
        else:
            stale += 1
        write_json(out/'training.json', records)
        if fixed_epochs is None and stale >= config.get('patience', 3):
            break
    winner = records[-1] if fixed_epochs is not None else max(records, key=lambda r: r['auc'])
    if final:
        p = predict(load_model(out, device), evaluation, valid, v, device)
        np.save(out/'evaluation_predictions.npy', p)
    result = dict(name=name, cutoff=cutoff, method='native_history_attention', config=config,
                  augmentation_config={'native_attention': config, 'feature_mode': 'spectral_multi_pair'},
                  best_epoch=winner['epoch'], **{k: a for k, a in winner.items() if k not in ['epoch', 'seconds']},
                  training_events=len(fit), anchor_cutoff=anchor_cutoff, base_path=str(base_path.relative_to(ROOT)),
                  base_sha256=sha(base_path/'model.pt'), parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  seconds=time.monotonic()-started)
    write_json(out/'result.json', result)
    return result


def train(name, cutoff, fixed_epochs, overrides):
    from arena_ood import arrays
    from arena_ood_features import dataset
    from ctr_data import load_cache
    config = overrides['native_attention']
    before = guard()
    out = OUT/'runs'/name/cutoff
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'result.json').exists():
            result = json.loads((out/'result.json').read_text())
            assert result['config'] == config
            return result
        base, info, labels, x, v = arrays(cutoff, 'spectral_multi')
        base, info, labels, x, v = dataset(cutoff, base, labels, x, v, info, 'pair_memory')
        _, raw = load_cache(ROOT/'data/cache_v1', 'train')
        base_path = OUT/'runs/spectral_multi_wide'/cutoff
        assert base_path.joinpath('model.pt').exists()
        result = train_loop(name, cutoff, config, raw, labels['train_indices'], raw, labels['validation_indices'],
                            x, v, base_path, base['feature_config']['train_start'], out, fixed_epochs)
        result.update(feature_config=base['feature_config'], feature_dataset=base['feature_dataset'],
                      augmentation_config=overrides, feature_names=info['feature_names'])
        assert guard() == before
        write_json(out/'result.json', result)
        return result


def fit_final(name, final):
    from arena_ood_export import final_inputs
    r = json.loads((OUT/'runs'/name/'2025-01-26/result.json').read_text())
    out = final/'models'/name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'recipe.json').exists():
        return out
    base, raw, evaluation, fit, x, v, names = final_inputs('spectral_multi_pair')
    base_path = ROOT/'competition/final_v6/models/spectral_multi_wide'
    result = train_loop(name, '2025-01-27', r['config'], raw, fit, evaluation, np.arange(len(evaluation['users'])),
                        x, v, base_path, '2025-01-24', out, r['best_epoch'], final=True)
    write_json(out/'recipe.json', dict(name=name, native_attention=True, model_config=r['config'],
               augmentation_config=r['augmentation_config'], feature_config=base['feature_config'],
               feature_names=names, training_events=len(fit), fixed_epochs=r['best_epoch'],
               training_indices_sha256=hashlib.sha256(fit.tobytes()).hexdigest(), result=result))
    return out
