"""Causal exposure histories: strict query-time AND held-out-cutoff exclusion.

All simultaneous observations are grouped. Each timeline entry is the state
AFTER that timestamp; queries select timestamps strictly before min(query, cap).
Item/global statistics therefore cannot consume another validation user's label.
"""
import fcntl
import functools
import hashlib
import json
import os
import time
import numpy as np
from numba import njit, prange

VERSION = 1
USER_TAU = np.array([6., 24., 72.]) * 3600
ITEM_TAU = np.array([24., 72.]) * 3600
COMMON = ['known', 'log_exposures', 'log_clicks', 'weak_logit', 'strong_logit',
          'log_gap_exposure', 'log_gap_click', 'log_gap_nonclick', 'signed_streak',
          'last_click_fraction', 'last3_logit', 'last5_logit', 'last10_logit', 'log_activity']


@njit(parallel=True, cache=True)
def _states(ptr, ts, count, clicks, taus):
    size = len(ts)
    total = np.zeros(size, np.int32)
    positive = np.zeros(size, np.int32)
    last_c = np.zeros(size, np.int64)
    last_n = np.zeros(size, np.int64)
    streak = np.zeros(size, np.int32)
    mass = np.zeros((size, len(taus)), np.float32)
    clicked_mass = np.zeros_like(mass)
    for uid in prange(len(ptr)-1):
        n, c, lc, ln, run, previous = 0, 0, 0, 0, 0, 0
        a, b = ptr[uid], ptr[uid+1]
        for p in range(a, b):
            n += count[p]
            c += clicks[p]
            lc = ts[p] if clicks[p] > 0 else lc
            ln = ts[p] if clicks[p] < count[p] else ln
            yes = clicks[p] > 0
            run = max(run, 0)+1 if yes else min(run, 0)-1
            total[p], positive[p] = n, c
            last_c[p], last_n[p], streak[p] = lc, ln, run
            for k in range(len(taus)):
                decay = np.exp(-(ts[p]-previous)/taus[k]) if p > a else 0.
                mass[p, k] = (mass[p-1, k]*decay if p > a else 0.)+count[p]
                clicked_mass[p, k] = (clicked_mass[p-1, k]*decay if p > a else 0.)+clicks[p]
            previous = ts[p]
    return total, positive, last_c, last_n, streak, mass, clicked_mass


def timeline(ids, ts, outcomes, n_ids, taus):
    ids, ts, outcomes = np.asarray(ids), np.asarray(ts), np.asarray(outcomes)
    assert len(ids) == len(ts) == len(outcomes)
    order = np.lexsort((ts, ids))
    ids, ts, outcomes = ids[order], ts[order], outcomes[order]
    first = np.r_[0, np.flatnonzero((ids[1:] != ids[:-1]) | (ts[1:] != ts[:-1]))+1] if len(ids) else np.array([], dtype='i')
    group_id, group_ts = ids[first], ts[first].astype('int64')
    count = np.diff(np.r_[first, len(ids)]).astype('int32') if len(first) else np.array([], dtype='int32')
    clicked = np.add.reduceat(outcomes.astype('int32'), first) if len(first) else np.array([], dtype='int32')
    ptr = np.r_[0, np.cumsum(np.bincount(group_id, minlength=n_ids))].astype('int64')
    values = _states(ptr, group_ts, count, clicked, taus)
    return dict(zip(['total', 'clicks', 'last_c', 'last_n', 'streak', 'mass', 'clicked_mass'], values),
                ptr=ptr, ts=group_ts, group_count=count, group_clicks=clicked)


def build(raw, cap, n_users, n_items):
    # Filtering happens before any label access, including global/item aggregates.
    ix = np.flatnonzero(raw['timestamps'] < cap)
    ts = np.asarray(raw['timestamps'][ix])
    y = np.asarray(raw['labels'][ix])
    users, items = np.asarray(raw['users'][ix]), np.asarray(raw['items'][ix])
    assert np.all(y.sum(1) <= 1) and np.all(y <= 1)
    u = timeline(users, ts, y.sum(1), n_users, USER_TAU)
    i = timeline(items.ravel(), np.repeat(ts, items.shape[1]), y.ravel(), n_items, ITEM_TAU)
    g = timeline(np.zeros(len(ix), 'int32'), ts, y.sum(1), 1, np.array([], 'float64'))
    packed = {tag+'_'+k: v for tag, obj in [('u', u), ('i', i), ('g', g)] for k, v in obj.items()}
    return packed, dict(fitting_events=len(ix), max_fitting_timestamp=int(ts.max()) if len(ts) else None,
                        fitting_indices_sha256=hashlib.sha256(ix.tobytes()).hexdigest(), cap=int(cap))


@functools.lru_cache(maxsize=1)
def _load(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def memory(cutoff):
    from arena_core import snapshot
    from arena_ood import ROOT, write_json, sha
    from ctr_data import load_cache
    from run import timestamp
    folder = snapshot(cutoff)/('online_events_v'+str(VERSION))
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (folder/'result.json').exists():
            tick = time.monotonic()
            meta, raw = load_cache(ROOT/'data/cache_v1', 'train')
            packed, info = build(raw, timestamp(cutoff), meta['n_users'], meta['n_items'])
            tmp = folder/'index.tmp.npz'
            np.savez_compressed(tmp, **packed)
            os.replace(tmp, folder/'index.npz')
            write_json(folder/'result.json', dict(info, cutoff=cutoff, version=VERSION, cache_id=meta['cache_id'],
                       index_sha256=sha(folder/'index.npz'), seconds=time.monotonic()-tick,
                       policy='Only fitting labels strictly before cap; queries exclude simultaneous/current/future timestamps.'))
            print(json.dumps(dict(stage='online_events_index', cutoff=cutoff, **info)), flush=True)
        info = json.loads((folder/'result.json').read_text())
        assert info['version'] == VERSION and info['cap'] == timestamp(cutoff)
    return _load(str((folder/'index.npz').resolve()))


@njit(cache=True)
def _logit(p):
    p = min(max(p, 1e-6), 1-1e-6)
    return np.log(p)-np.log1p(-p)


@njit(parallel=True, cache=True)
def _query(ids, query_ts, lookup_ts, priors, ptr, ts, total, clicks, last_c, last_n,
           streak, group_count, group_clicks, mass, clicked_mass, taus, weak, strong):
    out = np.zeros((len(ids), 14+2*len(taus)), np.float32)
    for row in prange(len(ids)):
        uid, q, limit, prior = ids[row], query_ts[row], lookup_ts[row], priors[row]
        a, b = 0, 0
        if 0 < uid < len(ptr)-1:
            a, b = ptr[uid], ptr[uid+1]
        lo, hi = a, b
        while lo < hi:
            mid = (lo+hi)//2
            if ts[mid] < limit:
                lo = mid+1
            else:
                hi = mid
        p = lo-1
        known = p >= a
        n, c = (total[p], clicks[p]) if known else (0, 0)
        out[row, 0], out[row, 1], out[row, 2] = known, np.log1p(n), np.log1p(c)
        out[row, 3] = _logit((c+weak*prior)/(n+weak))
        out[row, 4] = _logit((c+strong*prior)/(n+strong))
        for k in range(3):
            t = (ts[p] if k == 0 else last_c[p] if k == 1 else last_n[p]) if known else 0
            gap = min(max((q-t)/3600., 0), 720.) if t > 0 else 720.
            out[row, 5+k] = np.log1p(gap)
        if known:
            run = streak[p]
            out[row, 8] = np.sign(run)*np.log1p(abs(run))
            out[row, 9] = group_clicks[p]/group_count[p]
            out[row, 13] = np.log1p(n/max((q-ts[a])/86400., 1/24.))
        for k in range(3):
            window = 3 if k == 0 else 5 if k == 1 else 10
            previous = p-window
            wn = n-(total[previous] if previous >= a and known else 0)
            wc = c-(clicks[previous] if previous >= a and known else 0)
            out[row, 10+k] = _logit((wc+weak*prior)/(wn+weak))
        for k in range(len(taus)):
            decay = np.exp(-max(q-ts[p], 0)/taus[k]) if known else 0.
            dn = mass[p, k]*decay if known else 0.
            dc = clicked_mass[p, k]*decay if known else 0.
            out[row, 14+2*k] = np.log1p(dn)
            out[row, 15+2*k] = _logit((dc+weak*prior)/(dn+weak))
    return out


def features(index, data, indices, cap):
    """Pure query operation: never accesses the query labels or source arrays."""
    users, items = np.asarray(data['users'][indices]), np.asarray(data['items'][indices])
    q = np.asarray(data['timestamps'][indices])
    limit = np.minimum(q, cap)
    gp = np.searchsorted(index['g_ts'], limit, side='left')-1
    prior = np.full(len(q), .6)
    ok = gp >= 0
    prior[ok] = (index['g_clicks'][gp[ok]]+6)/(index['g_total'][gp[ok]]+10)

    def query(tag, ids, qt, lim, pri, taus, weak, strong):
        keys = ['ptr', 'ts', 'total', 'clicks', 'last_c', 'last_n', 'streak', 'group_count',
                'group_clicks', 'mass', 'clicked_mass']
        return _query(ids, qt, lim, pri, *[index[tag+'_'+k] for k in keys], taus, weak, strong)

    u = query('u', users, q, limit, prior, USER_TAU, 5., 20.)
    width = items.shape[1]
    i = query('i', items.ravel(), np.repeat(q, width), np.repeat(limit, width),
              np.repeat(prior/width, width), ITEM_TAU, 50., 200.)
    repeated_u = np.repeat(u, width, axis=0)
    from scipy.special import expit, logit
    row_prior = np.repeat(prior/width, width).clip(1e-6, 1-1e-6)
    online_user_row = expit(repeated_u[:, 3])/width
    combined = logit(online_user_row.clip(1e-6, 1-1e-6))+i[:, 3]-logit(row_prior)
    relative = combined.reshape(-1, width)
    relative = (relative-relative.mean(1, keepdims=True)).ravel()
    # tau24 is the second user decay (column17), first item decay (15).
    extra = np.stack([logit(row_prior), combined, relative,
                      repeated_u[:, 17]-repeated_u[:, 3], i[:, 15]-i[:, 3]], 1)
    names = ['online_user_'+n for n in COMMON]+['online_user_'+str(int(t/3600))+'h_'+n for t in USER_TAU for n in ['log_mass', 'logit']]
    names += ['online_item_'+n for n in COMMON]+['online_item_'+str(int(t/3600))+'h_'+n for t in ITEM_TAU for n in ['log_mass', 'logit']]
    names += ['online_global_logit', 'online_combined_logit', 'online_combined_relative', 'online_user_24h_delta', 'online_item_24h_delta']
    result = np.concatenate([repeated_u, i, extra], 1).astype('float32')
    assert result.shape == (len(indices)*width, 43) and np.isfinite(result).all()
    return result, names


def extra_features(data, indices, cutoff, config=None):
    from run import timestamp
    cap = (config or {}).get('online_cap', cutoff)
    index = memory(cap)
    pieces, names = [], []
    for start in range(0, len(indices), 8192):
        x, names = features(index, data, indices[start:start+8192], timestamp(cap))
        pieces.append(x)
    return np.concatenate(pieces), names
