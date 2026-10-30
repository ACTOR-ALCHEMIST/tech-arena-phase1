"""Delayed first-query training from multiple strictly earlier snapshot origins."""
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
from arena_ood import ROOT, write_json
from arena_daily import EventSubset, StackRows
from arena_ood_features import FeatureMatrix

OUT = ROOT/'competition/push084/origins'
DELAY_NAME = 'origin_log_delay_hours'


def delay(data, indices, origin):
    from run import timestamp
    elapsed = (data['timestamps'][indices]-timestamp(origin))/3600
    assert elapsed.min() >= 0
    return np.repeat(np.log1p(elapsed)/np.log1p(168), 10).astype('f')[:, None]


def part(origin, cutoff, extension):
    from arena_ood import arrays, source
    from arena_features import complete_features
    from arena_ood_features import extension_features
    from ctr_data import load_cache, temporal_split
    from run import timestamp
    _, data = load_cache(ROOT/'data/cache_v1', 'train')
    folder = OUT/origin
    folder.mkdir(parents=True, exist_ok=True)
    if origin in ['2025-01-24', '2025-01-25', '2025-01-26']:
        base, info, labels, _, values = arrays(origin)
        indices, names = labels['validation_indices'], info['feature_names']
        fc = base['feature_config']
    else:
        fc = source('2025-01-26')['feature_config']
        _, indices = temporal_split(data, timestamp(origin))
        if not (folder/'base.json').exists() or not (folder/'base.npy').exists():
            x, names = complete_features(data, indices, origin, fc)
            np.save(folder/'base.npy', x)
            write_json(folder/'base.json', dict(names=names, indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest()))
        meta = json.loads((folder/'base.json').read_text())
        assert meta['indices_sha256'] == hashlib.sha256(indices.tobytes()).hexdigest()
        values, names = np.load(folder/'base.npy', mmap_mode='r'), meta['names']
    if extension:
        tag = extension if isinstance(extension, str) else 'rich'
        if not (folder/(tag+'.json')).exists() or not (folder/(tag+'.npy')).exists():
            x, cols = extension_features(data, indices, origin, fc, extension)
            np.save(folder/(tag+'.npy'), x)
            write_json(folder/(tag+'.json'), dict(names=cols, indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest()))
        meta = json.loads((folder/(tag+'.json')).read_text())
        assert meta['indices_sha256'] == hashlib.sha256(indices.tobytes()).hexdigest()
        values, names = FeatureMatrix(values, np.load(folder/(tag+'.npy'), mmap_mode='r')), names+meta['names']
    positions = np.flatnonzero(data['timestamps'][indices] < timestamp(cutoff))
    ix = indices[positions]
    assert data['timestamps'][ix].min() >= timestamp(origin)
    assert data['timestamps'][ix].max() < timestamp(cutoff)
    assert len(np.unique(data['users'][ix])) == len(ix)
    # A user's first event after origin cannot have clicks from later events.
    h, age = data['history'][ix], data['age_hours'][ix]
    click_minutes = data['timestamps'][ix, None]//60-np.rint(age*60).astype(np.int64)
    assert np.all(click_minutes[h != 0] < timestamp(origin)//60)
    values = FeatureMatrix(EventSubset(values, positions), delay(data, ix, origin))
    return values, ix, names+[DELAY_NAME]


def training_inputs(cutoff, extension=False, days=3):
    from run import timestamp
    end = timestamp(cutoff)
    pieces, indices, names = [], [], None
    for offset in range(days, 0, -1):
        origin = datetime.fromtimestamp(end-offset*86400, timezone.utc).strftime('%Y-%m-%d')
        x, ix, cols = part(origin, cutoff, extension)
        if names is not None:
            assert names == cols
        pieces.append(x)
        indices.append(ix)
        names = cols
    ix = np.concatenate(indices)
    # The same observed target can be viewed from multiple older information
    # sets. These views never cross the fitting/validation boundary.
    print(json.dumps(dict(stage='delayed_first_queries', cutoff=cutoff, origins=days, views=len(ix), unique_events=len(np.unique(ix)))), flush=True)
    return StackRows(*pieces), ix, names


def dataset(cutoff, extension=False, days=3):
    from arena_ood import arrays
    from ctr_data import load_cache
    base, info, labels, _, v = arrays(cutoff, extension)
    x, ix, names = training_inputs(cutoff, extension, days)
    _, data = load_cache(ROOT/'data/cache_v1', 'train')
    assert names == info['feature_names']+[DELAY_NAME]
    assert not np.intersect1d(ix, labels['validation_indices']).size
    valid = FeatureMatrix(v, delay(data, labels['validation_indices'], cutoff))
    labels = dict(labels, train=data['labels'][ix].ravel(), train_indices=ix)
    return base, dict(info, feature_names=names), labels, x, valid
