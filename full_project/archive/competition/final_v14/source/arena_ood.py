"""History corruption experiments; archived models and clean caches are immutable.

Augmented examples keep the complete ten-item slate and original label. Every
history-dependent feature is recomputed from compacted clicks and aligned ages.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.55')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'competition/ood'
RECIPES = {
    'thin15_p20': {'augmentation': {'mode': 'thin', 'drop_probability': .15}, 'mix': .20},
    'thin35_p35': {'augmentation': {'mode': 'thin', 'drop_probability': .35}, 'mix': .35},
    'short_p20': {'augmentation': {'mode': 'truncate', 'limits': [5, 10]}, 'mix': .20},
    'cold_p10': {'augmentation': {'mode': 'cold_user'}, 'mix': .10},
    'rich_clean': {'expanded': True, 'mix': 0},
    'rich_thin15': {'expanded': True, 'augmentation': {'mode': 'thin', 'drop_probability': .15}, 'mix': .20},
    'rich_wide': {'expanded': True, 'mix': 0, 'model': {'architecture': 'v2_wide', 'seed': 73}},
    'rich_bce_thin15': {'expanded': True, 'augmentation': {'mode': 'thin', 'drop_probability': .15},
                        'mix': .20, 'model': {'loss': 'bce'}},
    'consensus_clean': {'feature_mode': 'consensus', 'mix': 0},
    'consensus_thin15': {'feature_mode': 'consensus', 'augmentation': {'mode': 'thin', 'drop_probability': .15}, 'mix': .20},
    'context_clean': {'feature_mode': 'context', 'mix': 0},
    'context_thin15': {'feature_mode': 'context', 'augmentation': {'mode': 'thin', 'drop_probability': .15}, 'mix': .20},
    'rich_wide_thin15': {'expanded': True, 'augmentation': {'mode': 'thin', 'drop_probability': .15},
                         'mix': .20, 'model': {'architecture': 'v2_wide', 'seed': 73}},
    'wide_thin15': {'augmentation': {'mode': 'thin', 'drop_probability': .15},
                    'mix': .20, 'model': {'architecture': 'v2_wide', 'seed': 73}},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def guard():
    paths = ['competition/CURRENT_BEST.json', 'competition/final_v3/submission.csv',
             'competition/final_v3/manifest.json', 'competition/final_v3/acceptance.json',
             'competition/final/submission.csv', 'competition/final/manifest.json',
             'competition/final/acceptance.json']
    latest = json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    paths += [latest[k] for k in ['csv', 'manifest', 'acceptance']]
    return {p: sha(ROOT/p) for p in paths}


def corrupt(data, spec, seed=9143):
    """Return independent inputs. Does not read labels, IDs or candidate targets."""
    out = dict(data)
    history = np.array(data['history'], copy=True)
    ages = np.array(data['age_hours'], copy=True)
    keep = history != 0
    rng = np.random.default_rng(seed)
    mode = spec['mode']
    if mode == 'thin':
        p = spec['drop_probability']
        if not 0 <= p <= 1:
            raise ValueError('Invalid deletion probability')
        if p == 0:
            # A no-op must preserve the original floating point context exactly.
            # Rebuilding log-gap/span from rounded ages can move legacy tree
            # inputs by one ULP even when no click was removed.
            return dict(data, history=history, age_hours=ages, context=np.array(data['context'], copy=True))
        keep &= rng.random(history.shape) >= p
    elif mode == 'truncate':
        limits = rng.choice(spec['limits'], len(history))
        if np.any((limits < 0) | (limits > history.shape[1])):
            raise ValueError('Invalid history limit')
        # Rank occupied positions, even if a caller supplies a sparse history.
        rank = np.cumsum(keep, axis=1)
        keep &= rank > (keep.sum(1) - limits)[:, None]
    elif mode == 'lag':
        count = rng.integers(1, spec.get('max_missing', 2)+1, size=len(history))
        keep &= np.cumsum(keep, axis=1) <= (keep.sum(1)-count)[:, None]
    elif mode == 'cold_user':
        out['users'] = np.zeros_like(data['users'])
    else:
        raise ValueError(mode)
    # All downstream "last item" features expect left-aligned, hole-free history.
    rows, cols = np.nonzero(keep)
    destination = np.cumsum(keep, axis=1)[rows, cols]-1
    new_h = np.zeros_like(history)
    new_a = np.zeros_like(ages)
    new_h[rows, destination] = history[rows, cols]
    new_a[rows, destination] = ages[rows, cols]
    count = keep.sum(1)
    latest = np.min(np.where(new_h != 0, new_a, np.inf), axis=1)
    latest = np.where(count > 0, latest, 0)
    oldest = np.max(np.where(new_h != 0, new_a, 0), axis=1)
    context = np.array(data['context'], copy=True)
    context[:, 3] = count/20
    context[:, 4] = np.log1p(latest)/np.log1p(24*30)
    context[:, 5] = np.log1p(np.maximum(oldest-latest, 0))/np.log1p(24*30)
    out.update(history=new_h, age_hours=new_a, context=context)
    return out


def source(cutoff):
    if cutoff == '2025-01-24':
        primary = source('2025-01-26')
        reference = json.loads((ROOT/'competition/experiments/v2_multiview_cross256_s2026_locked24/result.json').read_text())
        return dict(primary, cutoff=cutoff, feature_dataset=reference['feature_dataset'], feature_config=reference['feature_config'])
    name = 'v3_cross512_s2027' + ('' if cutoff == '2025-01-26' else '_locked'+cutoff[-2:])
    return json.loads((ROOT/'competition/experiments'/name/'result.json').read_text())


def arrays(cutoff, expanded=False):
    base = source(cutoff)
    folder = ROOT/base['feature_dataset']
    info = json.loads((folder/'metadata.json').read_text())
    labels = np.load(folder/'labels.npz')
    train, valid = np.load(folder/'train.npy', mmap_mode='r'), np.load(folder/'valid.npy', mmap_mode='r')
    if expanded:
        from arena_ood_features import dataset
        return dataset(cutoff, base, labels, train, valid, info, expanded)
    return base, info, labels, train, valid


def bank(cutoff, spec, fraction=.5, seed=61973, expanded=False):
    """Cache an input view for a fixed random half of events; mix anew per batch.

    Subsampling saves disk, not labelled training examples: every clean event is
    trained each epoch. Unselected events always use the original input.
    """
    from ctr_data import load_cache
    from arena_features import complete_features
    base, info, labels, x, _ = arrays(cutoff)
    meta, data = load_cache(ROOT/'data/cache_v1', 'train')
    spec = dict(spec)
    key = {'cutoff': cutoff, 'spec': spec, 'fraction': fraction, 'seed': seed,
           'cache_id': meta['cache_id'], 'feature_dataset': base['feature_dataset'],
           'train_indices_sha': hashlib.sha256(labels['train_indices'].tobytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    folder = OUT/'banks'/digest
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (folder/'metadata.json').exists() or not (folder/'features.npy').exists():
            count = round(len(labels['train_indices'])*fraction)
            selected = np.sort(np.random.default_rng(seed).choice(len(labels['train_indices']), count, replace=False))
            indices = labels['train_indices'][selected]
            inputs = {k: np.array(v[indices]) for k, v in data.items() if k != 'labels'}
            inputs = corrupt(inputs, spec, seed+1)
            print(json.dumps({'stage': 'augmented_features', 'cutoff': cutoff, 'spec': spec, 'events': count}), flush=True)
            features, columns = complete_features(inputs, np.arange(count), base['feature_config']['train_start'], base['feature_config'])
            assert columns == info['feature_names']
            assert features.shape == (count*10, x.shape[1]) and np.isfinite(features).all()
            np.save(folder/'features.npy', features)
            np.save(folder/'selected.npy', selected)
            write_json(folder/'metadata.json', dict(key, events=count, feature_names=columns,
                       feature_snapshot=base['feature_config']['train_start'],
                       label_policy='Original labels and all ten candidates retained; corruption reads no labels.'))
    selected = np.load(folder/'selected.npy')
    lookup = np.full(len(labels['train_indices']), -1, dtype=np.int64)
    lookup[selected] = np.arange(len(selected))
    values = np.load(folder/'features.npy', mmap_mode='r')
    if expanded:
        from arena_ood_features import extension_features, FeatureMatrix
        tag = expanded+'_extra' if isinstance(expanded, str) else 'extra'
        if not (folder/(tag+'.json')).exists() or not (folder/(tag+'.npy')).exists():
            indices = labels['train_indices'][selected]
            inputs = {k: np.array(v[indices]) for k, v in data.items() if k != 'labels'}
            inputs = corrupt(inputs, spec, seed+1)
            extra, columns = extension_features(inputs, np.arange(len(indices)), base['feature_config']['train_start'], base['feature_config'], expanded)
            np.save(folder/(tag+'.npy'), extra)
            write_json(folder/(tag+'.json'), {'names': columns, 'events': len(indices)})
        values = FeatureMatrix(values, np.load(folder/(tag+'.npy'), mmap_mode='r'))
    return values, lookup, folder


def train(name, cutoff='2025-01-26', fixed_epochs=None, overrides=None):
    from arena_tabular import predict
    from arena_push_model import build_model, normalizer
    from arena_core import evaluate
    from arena_validation import importance_weights, weighted_auc
    before = guard()
    config = dict(RECIPES[name] if name in RECIPES else {})
    if overrides:
        config.update(overrides)
    if not config:
        raise ValueError(name)
    if config.get('native_attention'):
        from arena_native_attention import train as native_train
        return native_train(name, cutoff, fixed_epochs, config)
    if config.get('graph_tree'):
        from arena_graph_trees import train as tree_train
        return tree_train(name, cutoff, fixed_epochs, config)
    if config.get('graph_lgbm'):
        from arena_graph_lgbm import train as tree_train
        return tree_train(name, cutoff, fixed_epochs, config)
    if config.get('bayes_mf'):
        from arena_bayes_mf import train as bayes_train
        return bayes_train(name, cutoff, fixed_epochs, config)
    out = OUT/'runs'/name/cutoff
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'result.json').exists():
            result = json.loads((out/'result.json').read_text())
            assert result['augmentation_config'] == config, 'Use a new name for changed experiments'
            return result
        extension = config.get('feature_mode', config.get('expanded', False))
        from arena_windows import days
        origin_days, rolling_days = days(config, cutoff, 'origin'), days(config, cutoff, 'rolling')
        if origin_days:
            from arena_origins import dataset as origin_dataset
            base, info, labels, x, v = origin_dataset(cutoff, extension, origin_days)
        elif rolling_days:
            from arena_daily import dataset as daily_dataset
            base, info, labels, x, v = daily_dataset(cutoff, extension, rolling_days)
        else:
            base, info, labels, x, v = arrays(cutoff, expanded=extension)
        cfg = copy.deepcopy(base['config'])
        cfg.update(epochs=25, lr_schedule_epochs=40, fixed_epochs=False)
        cfg.update(config.get('model', {}))
        if config.get('drop_temporal'):
            from arena_push_model import temporal_columns
            cfg['input_mask_columns'] = temporal_columns(info['feature_names'], config['drop_temporal'])
            cfg['dropped_feature_names'] = [info['feature_names'][i] for i in cfg['input_mask_columns']]
        cfg['name'] = name
        torch.manual_seed(cfg['seed'])
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        augmented = None
        if config.get('mix', 0):
            assert not (rolling_days or origin_days), 'Rolling augmentation needs its own row-aligned bank'
            augmented, lookup, bank_path = bank(cutoff, config['augmentation'], expanded=extension)
            conditional = config['mix']/(np.count_nonzero(lookup >= 0)/len(lookup))
            assert 0 <= conditional <= 1
        mean, scale = normalizer(x, cfg)
        model = build_model(x.shape[1], mean, scale, cfg).to(device)
        ema = copy.deepcopy(model).eval() if cfg.get('ema') else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['lr_schedule_epochs'], eta_min=.0001)
        rng = np.random.default_rng(cfg['seed'])
        augmentation_rng = np.random.default_rng(cfg['seed']+7331)
        y = labels['train'].reshape(-1, 10)
        vy = labels['valid'].reshape(-1, 10)
        from arena_objectives import loss as objective, user_weights
        if cfg.get('user_balance'):
            from ctr_data import load_cache
            _, raw = load_cache(ROOT/'data/cache_v1', 'train')
            event_weights = user_weights(raw['users'][labels['train_indices']], cfg['user_balance'])
        else:
            event_weights = None
        iw, profile = importance_weights(cutoff, labels['validation_indices'])
        write_json(out/'config.json', {'model': cfg, 'augmentation': config, 'source': base['name']})
        records, best, stale = [], -1., 0
        started = time.monotonic()
        epochs = fixed_epochs if fixed_epochs is not None else cfg['epochs']
        for epoch in range(1, epochs+1):
            tick = time.monotonic()
            model.train()
            order = rng.permutation(len(y))
            summed, corrupted_events = 0., 0
            for offset in range(0, len(order), cfg.get('batch_events', 1024)):
                chosen = order[offset:offset+cfg.get('batch_events', 1024)]
                rows = (chosen[:, None]*10+np.arange(10)).ravel()
                values = np.array(x[rows]).reshape(-1, 10, x.shape[1])
                if augmented is not None:
                    positions = lookup[chosen]
                    selected = (positions >= 0) & (augmentation_rng.random(len(chosen)) < conditional)
                    if selected.any():
                        aug_rows = (positions[selected, None]*10+np.arange(10)).ravel()
                        values[selected] = augmented[aug_rows].reshape(-1, 10, x.shape[1])
                        corrupted_events += int(selected.sum())
                bx = torch.tensor(values, device=device)
                by = torch.tensor(y[chosen], dtype=torch.float32, device=device)
                z = model(bx)
                bw = torch.tensor(event_weights[chosen], device=device) if event_weights is not None else None
                loss = objective(z, by, cfg, bw)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                summed += loss.item()*len(chosen)
                if ema is not None:
                    with torch.no_grad():
                        for target, parameter in zip(ema.parameters(), model.parameters()):
                            target.mul_(cfg['ema']).add_(parameter, alpha=1-cfg['ema'])
            scheduler.step()
            prediction = predict(ema if ema is not None else model, v, device)
            row = dict(epoch=epoch, train_loss=summed/len(y), **evaluate(vy, prediction),
                       weighted_auc=weighted_auc(vy, prediction, iw), augmented_events=corrupted_events,
                       seconds=time.monotonic()-tick)
            records.append(row)
            print(json.dumps(dict(name=name, cutoff=cutoff, **row)), flush=True)
            save = epoch==epochs if fixed_epochs is not None else row['auc']>best
            if save:
                best, stale = row['auc'], 0
                torch.save({'state_dict': (ema if ema is not None else model).state_dict(),
                            'width': x.shape[1], 'loss': cfg['loss'], 'config': cfg}, out/'model.pt')
                np.savez_compressed(out/'validation.npz', event_indices=labels['validation_indices'], labels=vy, predictions=prediction)
            else:
                stale += 1
            write_json(out/'training.json', records)
            if fixed_epochs is None and stale >= cfg.get('patience', 5):
                break
        winner = records[-1] if fixed_epochs is not None else max(records, key=lambda row: row['auc'])
        result = dict(name=name, cutoff=cutoff, method='ood_tabular', **{k: v for k, v in winner.items() if k != 'seconds'},
                      best_epoch=winner['epoch'], config=cfg, augmentation_config=config,
                      feature_config=base['feature_config'], feature_dataset=base['feature_dataset'],
                      source_experiment=base['name'], parameters=sum(p.numel() for p in model.parameters()),
                      seconds=time.monotonic()-started, cohort_profile=profile)
        # Individual fits use fixed data/configuration, not incumbent scores.
        # Permit another authorized campaign to advance the pointer while all
        # artifacts that existed at fitting start remain byte-for-byte intact.
        assert all(sha(ROOT/p) == digest for p, digest in before.items()
                   if p != 'competition/CURRENT_BEST.json'), 'Incumbent artifacts changed'
        write_json(out/'result.json', result)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['train'])
    p.add_argument('--name', required=True)
    p.add_argument('--cutoff', default='2025-01-26')
    p.add_argument('--fixed-epochs', type=int)
    p.add_argument('--config')
    a = p.parse_args()
    train(a.name, a.cutoff, a.fixed_epochs, json.loads(Path(a.config).read_text()) if a.config else None)
