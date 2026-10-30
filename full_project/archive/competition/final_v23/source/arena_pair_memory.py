"""Past-only exact user/item feedback and shrunken repeat-choice propensities."""
import fcntl
import numpy as np
from arena_core import snapshot
from arena_ood import write_json


def lookup(keys, values, query):
    if not len(keys):
        return np.zeros(np.asarray(query).shape, dtype=values.dtype)
    p = np.searchsorted(keys, query)
    safe = np.minimum(p, len(keys)-1)
    return np.where((p < len(keys)) & (keys[safe] == query), values[safe], 0)


def memory(cutoff):
    from ctr_data import load_cache
    from run import timestamp
    from arena_feedback import feedback
    folder = snapshot(cutoff)/'pair_memory_v1'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            return dict(np.load(folder/'memory.npz'))
        meta, data = load_cache('data/cache_v1', 'train')
        fit = np.flatnonzero(data['timestamps'] < timestamp(cutoff))
        nu, ni = meta['n_users'], meta['n_items']
        uid, iid = np.array(data['users'][fit]), np.array(data['items'][fit])
        labels = np.array(data['labels'][fit])
        ts = np.array(data['timestamps'][fit])//60
        keys = (uid[:, None].astype(np.int64)*ni+iid).ravel()
        ek, inv, count = np.unique(keys, return_inverse=True, return_counts=True)
        clicks = np.bincount(inv, weights=labels.ravel(), minlength=len(ek)).astype('i')
        last = np.zeros(len(ek), np.int32)
        np.maximum.at(last, inv, np.repeat(ts, 10).astype('i'))
        assert np.all(clicks <= count) and int(ts.max())*60 < timestamp(cutoff)
        del keys, inv
        ur_n, ur_c, un_n, un_c = [np.zeros(nu) for _ in range(4)]
        ir_n, ir_c = np.zeros(ni), np.zeros(ni)
        for start in range(0, len(fit), 32768):
            stop = min(len(fit), start+32768)
            rep = (iid[start:stop, :, None] == data['history'][fit[start:stop], None, :]).any(2)
            y = labels[start:stop]
            users = uid[start:stop]
            ur_n += np.bincount(users, weights=rep.sum(1), minlength=nu)
            ur_c += np.bincount(users, weights=(rep*y).sum(1), minlength=nu)
            un_n += np.bincount(users, weights=(~rep).sum(1), minlength=nu)
            un_c += np.bincount(users, weights=((~rep)*y).sum(1), minlength=nu)
            ir_n += np.bincount(iid[start:stop][rep], minlength=ni)
            ir_c += np.bincount(iid[start:stop][rep], weights=y[rep], minlength=ni)
        gr, gn = ur_c.sum()/max(ur_n.sum(), 1), un_c.sum()/max(un_n.sum(), 1)
        obs = feedback(cutoff)
        ck, cinv, cc = np.unique(obs['users'].astype(np.int64)*ni+obs['items'], return_inverse=True, return_counts=True)
        clast = np.zeros(len(ck), np.int32)
        np.maximum.at(clast, cinv, obs['minutes'])
        cu, ci = ck//ni, ck%ni
        totals = np.bincount(cu, weights=cc, minlength=nu)
        distinct = np.bincount(cu, minlength=nu)
        repeats = np.maximum(totals-distinct, 0)
        ir = np.bincount(ci, weights=np.maximum(cc-1, 0), minlength=ni)
        it = np.bincount(ci, weights=cc, minlength=ni)
        global_repeat = repeats.sum()/max(totals.sum(), 1)
        values = dict(exposure_keys=ek, exposure_count=count.astype('i'), exposure_clicks=clicks, exposure_last=last,
                      click_keys=ck, click_count=cc.astype('i'), click_last=clast,
                      user_repeat_ctr=((ur_c+5*gr)/(ur_n+5)).astype('f'),
                      user_nonrepeat_ctr=((un_c+20*gn)/(un_n+20)).astype('f'),
                      item_repeat_ctr=((ir_c+20*gr)/(ir_n+20)).astype('f'),
                      user_repeat_fraction=((repeats+5*global_repeat)/(totals+5)).astype('f'),
                      item_repeat_fraction=((ir+10*global_repeat)/(it+10)).astype('f'),
                      user_concentration=(np.bincount(cu, weights=cc.astype(float)**2, minlength=nu)/totals.clip(1)**2).astype('f'),
                      user_clicks=totals.astype('f'))
        np.savez_compressed(folder/'memory.npz', **values)
        write_json(folder/'result.json', dict(cutoff=cutoff, fitting_events=len(fit), unique_exposures=len(ek),
                   unique_clicked_pairs=len(ck), max_fitting_timestamp=int(ts.max())*60,
                   max_click_timestamp=int(obs['minutes'].max())*60,
                   label_policy='CTR numerators use fitting exposure labels only; warmup click counts are separate features.'))
        print(dict(stage='pair_memory', cutoff=cutoff, fitting_events=len(fit), unique_exposures=len(ek)), flush=True)
    return values


def extra_features(data, indices, cutoff, config=None):
    from ctr_data import load_cache
    from arena_core import sigmoid
    from run import timestamp
    assert np.all(data['timestamps'][indices] >= timestamp(cutoff)), 'Pair features cannot use information newer than a query'
    meta, _ = load_cache('data/cache_v1', 'train')
    m = memory(cutoff)
    stats = dict(np.load(snapshot(cutoff)/'stats.npz'))
    names = ['pair_'+n for n in ['ever_clicked', 'log_click_count', 'log_exposure_count', 'log_exposure_clicks',
             'log_nonclicks', 'posterior_logit_delta', 'seen_recency72h', 'clicked_recency168h',
             'clicked_outside_visible', 'user_repeat_ctr', 'user_nonrepeat_ctr', 'item_repeat_ctr',
             'user_repeat_fraction', 'user_concentration', 'user_log_clicks', 'item_repeat_fraction']]
    pieces = []
    for start in range(0, len(indices), 4096):
        ix = indices[start:start+4096]
        uid, iid = data['users'][ix], data['items'][ix]
        keys = uid[:, None].astype(np.int64)*meta['n_items']+iid
        ni = lookup(m['exposure_keys'], m['exposure_count'], keys)
        nc = lookup(m['exposure_keys'], m['exposure_clicks'], keys)
        cp = lookup(m['click_keys'], m['click_count'], keys)
        seen = lookup(m['exposure_keys'], m['exposure_last'], keys)
        clicked = lookup(m['click_keys'], m['click_last'], keys)
        current = data['timestamps'][ix, None]//60
        assert np.all(current[0] >= 0)
        sr = np.exp(-np.maximum(current-seen, 0)/(72*60))*(seen > 0)
        cr = np.exp(-np.maximum(current-clicked, 0)/(168*60))*(clicked > 0)
        rep = (iid[:, :, None] == data['history'][ix, None, :]).any(2)
        prior = float(np.log(stats['ctr']/(1-stats['ctr'])))+stats['ub'][uid, None]+stats['ib'][iid]
        probability = (nc+5*sigmoid(prior))/(ni+5)
        delta = np.log(probability.clip(1e-6, 1-1e-6)/(1-probability.clip(1e-6, 1-1e-6)))-prior
        expand = lambda x: np.broadcast_to(x[:, None], iid.shape)
        values = [(cp > 0), np.log1p(cp), np.log1p(ni), np.log1p(nc), np.log1p(ni-nc), delta, sr, cr,
                  (cp > 0)&~rep, m['user_repeat_ctr'][uid, None]*rep, m['user_nonrepeat_ctr'][uid, None]*(~rep),
                  m['item_repeat_ctr'][iid]*rep, expand(m['user_repeat_fraction'][uid]), expand(m['user_concentration'][uid]),
                  expand(np.log1p(m['user_clicks'][uid])), m['item_repeat_fraction'][iid]]
        pieces.append(np.stack(values, 2).astype('f'))
    x = np.concatenate(pieces).reshape(-1, len(names))
    assert np.isfinite(x).all()
    return x, names
