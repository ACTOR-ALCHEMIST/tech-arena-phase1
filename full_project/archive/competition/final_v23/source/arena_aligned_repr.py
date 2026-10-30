"""Fresh latent coordinates aligned to a fixed, strictly earlier reference.

Both user and item vectors receive the same orthogonal transform. All products
are preserved; the fixed reference makes individual coordinates comparable
between fitting/inference dates. Final snapshot caches are self-contained.
"""
import fcntl
import json
import os
import time
import numpy as np
from arena_core import snapshot
from arena_ood import ROOT, sha, write_json

RECIPE = dict(version=1, reference='2025-01-21', dimension=8, spherical_epoch=32,
              weighting='minimum_popularity_capped20', translation=False)
TAG = 'aligned_sphere_v1_to21'


def rotation(fresh_items, reference_items, weights):
    weights = np.asarray(weights, 'float64')
    assert fresh_items.shape == reference_items.shape and len(weights) == len(fresh_items)
    assert (weights >= 0).all() and (weights > 0).sum() >= fresh_items.shape[1]
    cross = (fresh_items.astype('float64')*weights[:, None]).T@reference_items.astype('float64')
    left, _, right = np.linalg.svd(cross, full_matrices=False)
    return left@right


def model(cutoff):
    if cutoff <= RECIPE['reference']:
        raise ValueError('Coordinate reference must precede the feature frame')
    path = snapshot(cutoff)
    folder = path/TAG
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            assert json.loads((folder/'result.json').read_text())['recipe'] == RECIPE
            # Do not load the reference snapshot here: a frozen final snapshot
            # must be sufficient to reproduce the submitted predictions.
            return dict(np.load(folder/'factors.npz'))
        from arena_spherical_repr import model as spherical
        from run import timestamp
        tick = time.monotonic()
        reference = RECIPE['reference']
        fresh_path, ref_path = spherical(cutoff), spherical(reference)
        fresh_file, ref_file = fresh_path/'factors_e32.npz', ref_path/'factors_e32.npz'
        fresh, anchor = dict(np.load(fresh_file)), dict(np.load(ref_file))
        a = dict(np.load(path/'stats.npz'))
        b = dict(np.load(snapshot(reference)/'stats.npz'))
        weights = np.minimum(np.minimum(a['pop'], b['pop']), 20)
        weights[0] = 0
        transform = rotation(fresh['items'], anchor['items'], weights)
        users, items = [(fresh[k].astype('float64')@transform).astype('float32') for k in ['users', 'items']]
        users[0] = items[0] = 0
        rep = dict(users=users, items=items, known_users=np.linalg.norm(users, axis=1)>0,
                   known_items=np.linalg.norm(items, axis=1)>0, rotation=transform)
        np.savez(folder/'factors.tmp.npz', **rep)
        os.replace(folder/'factors.tmp.npz', folder/'factors.npz')
        metadata = json.loads((path/'metadata.json').read_text())
        assert metadata['max_fit_timestamp'] < timestamp(cutoff)
        write_json(folder/'result.json', dict(recipe=RECIPE, cutoff=cutoff, cache_id=metadata['cache_id'],
                   max_fitting_timestamp=metadata['max_fit_timestamp'],
                   source_spherical_sha256=sha(fresh_file), reference_spherical_sha256=sha(ref_file),
                   common_items=int((weights > 0).sum()), orthogonality_max_error=float(np.abs(transform.T@transform-np.eye(8)).max()),
                   seconds=time.monotonic()-tick, policy='Fixed Jan21 reference; both representations strictly before query. No query labels used.'))
        print(json.dumps(dict(stage='aligned_coordinates', cutoff=cutoff, seconds=time.monotonic()-tick)), flush=True)
    return rep


def features(rep, data, indices):
    uid, iid = np.asarray(data['users'][indices]), np.asarray(data['items'][indices])
    hist, age = np.asarray(data['history'][indices]), np.asarray(data['age_hours'][indices])
    u, v, h = rep['users'][uid], rep['items'][iid], rep['items'][hist]
    visible = hist != 0
    known = visible & rep['known_items'][hist]
    length = visible.sum(1)
    h = h*known[..., None]
    mean = h.sum(1)/known.sum(1).clip(1)[:, None]
    last = known & (visible.cumsum(1) > (length-3)[:, None])
    recent = (h*last[..., None]).sum(1)/last.sum(1).clip(1)[:, None]
    weight = np.exp(-age/72)*known
    decay = (h*weight[..., None]).sum(1)/weight.sum(1).clip(1e-6)[:, None]
    expand = lambda x:np.broadcast_to(x[:, None], v.shape)
    values = [expand(u), v, u[:, None]*v, expand(mean), mean[:, None]*v,
              expand(recent), recent[:, None]*v, expand(decay), decay[:, None]*v]
    names = ['aligned_'+tag+'_'+str(k) for tag in ['user', 'item', 'user_item', 'history', 'history_item',
             'last3', 'last3_item', 'decay72', 'decay72_item'] for k in range(8)]
    reliability = np.stack([np.broadcast_to(rep['known_users'][uid, None], iid.shape), rep['known_items'][iid],
                           np.broadcast_to((known.sum(1)/length.clip(1))[:, None], iid.shape)], 2)
    result = np.concatenate(values+[reliability], 2).reshape(-1, 75).astype('float32')
    names += ['aligned_known_user', 'aligned_known_item', 'aligned_known_history_fraction']
    assert result.shape == (len(indices)*iid.shape[1], 75) and np.isfinite(result).all()
    return result, names


def extra_features(data, indices, cutoff, config=None):
    from run import timestamp
    assert np.all(data['timestamps'][indices] >= timestamp(cutoff)), 'Representation newer than query'
    rep = model(cutoff)
    pieces, names = [], []
    for start in range(0, len(indices), 4096):
        x, names = features(rep, data, indices[start:start+4096])
        pieces.append(x)
    return np.concatenate(pieces, 0), names


class LazyAlignedRows:
    """Generate identical float32 coordinates for requested flat candidate rows."""
    def __init__(self, rep, data, indices):
        self.rep = rep
        self.data = {k:data[k] for k in ['users', 'items', 'history', 'age_hours']}
        self.indices = np.asarray(indices)
        self.shape = (len(indices)*data['items'].shape[1], 75)
        self.width = data['items'].shape[1]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        if isinstance(index, slice):
            rows = np.arange(*index.indices(len(self)), dtype=np.int64)
        elif scalar:
            rows = np.array([index], dtype=np.int64)
        else:
            rows = np.asarray(index)
            rows = np.flatnonzero(rows) if rows.dtype == bool else rows.astype(np.int64, copy=False)
        rows = np.where(rows < 0, rows+len(self), rows)
        if np.any((rows < 0) | (rows >= len(self))):
            raise IndexError('Candidate row outside the feature matrix')
        if not len(rows):
            return np.empty((0, 75), np.float32)
        events, inverse = np.unique(rows//self.width, return_inverse=True)
        values = np.empty((len(events)*self.width, 75), np.float32)
        for start in range(0, len(events), 4096):
            stop = min(start+4096, len(events))
            block, _ = features(self.rep, self.data, self.indices[events[start:stop]])
            values[start*self.width:stop*self.width] = block
        result = values[inverse*self.width+rows%self.width]
        return result[0] if scalar else result


def lazy_features(data, indices, cutoff):
    from run import timestamp
    assert len(indices) and np.all(data['timestamps'][indices] >= timestamp(cutoff))
    rep = model(cutoff)
    _, names = features(rep, data, indices[:1])
    return LazyAlignedRows(rep, data, indices), names
