"""Train on first queries per user/day, each using that day's past-only frame."""
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
from arena_ood import ROOT, write_json
from arena_ood_features import FeatureMatrix

OUT = ROOT/'competition/push084/daily'


class EventSubset:
    def __init__(self, values, positions):
        self.values, self.positions = values, np.asarray(positions)
        self.shape = (len(positions)*10, values.shape[1])
    def __len__(self):
        return self.shape[0]
    def __getitem__(self, key):
        rows = np.arange(*key.indices(len(self))) if isinstance(key, slice) else np.asarray(key)
        return self.values[self.positions[rows//10]*10+rows%10]


class StackRows:
    def __init__(self, *parts):
        assert len({p.shape[1] for p in parts}) == 1
        self.parts = parts
        self.boundaries = np.r_[0, np.cumsum([len(p) for p in parts])]
        self.shape = (int(self.boundaries[-1]), parts[0].shape[1])
    def __len__(self):
        return self.shape[0]
    def __getitem__(self, key):
        rows = np.arange(*key.indices(len(self))) if isinstance(key, slice) else np.asarray(key)
        result = np.empty((len(rows), self.shape[1]), np.float32)
        for p, left, right in zip(self.parts, self.boundaries[:-1], self.boundaries[1:]):
            mask = (rows >= left) & (rows < right)
            if mask.any():
                result[mask] = p[rows[mask]-left]
        return result


def part(day, extension=False):
    from arena_ood import arrays, source
    from arena_features import complete_features
    from arena_ood_features import extension_features
    from ctr_data import load_cache, temporal_split
    from run import timestamp
    _, data = load_cache(ROOT/'data/cache_v1', 'train')
    folder = OUT/day
    folder.mkdir(parents=True, exist_ok=True)
    if day in ['2025-01-24', '2025-01-25', '2025-01-26']:
        base, info, labels, _, v = arrays(day)
        all_indices = labels['validation_indices']
        positions = np.flatnonzero(data['timestamps'][all_indices] < timestamp(day)+86400)
        indices = all_indices[positions]
        values = EventSubset(v, positions)
        names = info['feature_names']
        fc = base['feature_config']
    else:
        fc = source('2025-01-26')['feature_config']
        _, all_indices = temporal_split(data, timestamp(day))
        indices = all_indices[data['timestamps'][all_indices] < timestamp(day)+86400]
        if not (folder/'base.json').exists() or not (folder/'base.npy').exists():
            x, names = complete_features(data, indices, day, fc)
            np.save(folder/'base.npy', x)
            write_json(folder/'base.json', dict(names=names, indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest()))
        meta = json.loads((folder/'base.json').read_text())
        assert meta['indices_sha256'] == hashlib.sha256(indices.tobytes()).hexdigest()
        names = meta['names']
        values = np.load(folder/'base.npy', mmap_mode='r')
    assert np.all(data['timestamps'][indices] >= timestamp(day))
    assert np.all(data['timestamps'][indices] < timestamp(day)+86400)
    assert len(np.unique(data['users'][indices])) == len(indices)
    if extension:
        tag = extension if isinstance(extension, str) else 'rich'
        if not (folder/(tag+'.json')).exists() or not (folder/(tag+'.npy')).exists():
            x, cols = extension_features(data, indices, day, fc, extension)
            np.save(folder/(tag+'.npy'), x)
            write_json(folder/(tag+'.json'), dict(names=cols, indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest()))
        meta = json.loads((folder/(tag+'.json')).read_text())
        assert meta['indices_sha256'] == hashlib.sha256(indices.tobytes()).hexdigest()
        names = names+meta['names']
        values = FeatureMatrix(values, np.load(folder/(tag+'.npy'), mmap_mode='r'))
    write_json(folder/'protocol.json', dict(day=day, rows=len(indices)*10,
               first_event_per_user=True, min_timestamp=int(data['timestamps'][indices].min()),
               max_timestamp=int(data['timestamps'][indices].max()), feature_cutoff=day,
               indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest()))
    return values, indices, names


def training_inputs(cutoff, extension=False, days=3):
    from run import timestamp
    parts, indices, names = [], [], None
    end = timestamp(cutoff)
    for offset in range(days, 0, -1):
        day = datetime.fromtimestamp(end-offset*86400, timezone.utc).strftime('%Y-%m-%d')
        x, ix, cols = part(day, extension)
        if names is not None:
            assert names == cols
        names = cols
        parts.append(x)
        indices.append(ix)
    ix = np.concatenate(indices)
    assert len(np.unique(ix)) == len(ix), 'Daily fitting targets cannot repeat across days'
    print(json.dumps(dict(stage='daily_training_inputs', cutoff=cutoff, days=days, events=len(ix))), flush=True)
    return StackRows(*parts), ix, names


def dataset(cutoff, extension=False, days=3):
    from arena_ood import arrays
    from ctr_data import load_cache
    base, info, labels, _, v = arrays(cutoff, extension)
    x, ix, names = training_inputs(cutoff, extension, days)
    assert names == info['feature_names']
    _, data = load_cache(ROOT/'data/cache_v1', 'train')
    assert data['timestamps'][ix].max() < __import__('run').timestamp(cutoff)
    assert not np.intersect1d(ix, labels['validation_indices']).size
    labels = dict(labels)
    labels.update(train=data['labels'][ix].ravel(), train_indices=ix)
    return base, info, labels, x, v
