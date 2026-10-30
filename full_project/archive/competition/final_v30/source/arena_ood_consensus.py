"""Coordinate-aligned consensus of independent, past-only SGNS fits."""
import numpy as np


def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-6)


def extra_features(data, indices, cutoff, config):
    from arena_sgns_features import embedding_model
    from arena_core import snapshot
    size = 8
    first = embedding_model(cutoff, size, epochs=config['sgns_epochs']['8'], recipe=config['sgns_recipes']['8'])
    second_spec = config['sgns_extra'][0]
    second = embedding_model(cutoff, second_spec['factors'], recipe=second_spec['recipe'])
    stats = np.load(snapshot(cutoff)/'stats.npz')
    av, bv = unit(first['items']), unit(second['items'])
    au, bu = unit(first['users']), unit(second['users'])
    common = stats['pop'] > 0
    left, _, right = np.linalg.svd(bv[common].astype(np.float64).T@av[common].astype(np.float64))
    rotation = (left@right).astype(np.float32)
    bv, bu = bv@rotation, bu@rotation
    agreement_i, agreement_u = (av*bv).sum(1), (au*bu).sum(1)
    cv, cu = unit(av+bv), unit(au+bu)
    cv[~common] = 0
    cu[~stats['known_users']] = 0
    agreement_i[~common] = 0
    agreement_u[~stats['known_users']] = 0
    pieces = []
    for start in range(0, len(indices), 2048):
        ix = indices[start:start+2048]
        ids = data['items'][ix]
        users = data['users'][ix]
        hist = data['history'][ix]
        age = data['age_hours'][ix]
        mask = hist != 0
        length = mask.sum(1)
        v, h, u = cv[ids], cv[hist]*mask[..., None], cu[users]
        hc = np.einsum('bik,bjk->bij', v, h)
        cos = (v*u[:, None, :]).sum(2)
        values, names = [], []

        def add(name, value, relative=False):
            if value.ndim == 1:
                value = np.broadcast_to(value[:, None], ids.shape)
            values.append(value)
            names.append('consensus8_'+name)
            if relative:
                values.append(value-value.mean(1, keepdims=True))
                names.append('consensus8_'+name+'_relative')

        def weighted(weight):
            return (hc*weight[:, None, :]).sum(2)/weight.sum(1).clip(1e-6)[:, None]

        add('cos', cos, True)
        add('item_agreement', agreement_i[ids])
        add('user_agreement', agreement_u[users])
        mean = weighted(mask)
        add('hist_mean', mean, True)
        add('hist_max', np.where(mask[:, None, :], hc, -np.inf).max(2).clip(-1, 1)* (length>0)[:, None])
        add('recent72', weighted(np.exp(-age/72)*mask))
        add('recent24', weighted(np.exp(-age/24)*mask), True)
        rank = np.cumsum(mask, axis=1)
        add('last1', weighted(mask & (rank > (length-1)[:, None])))
        add('last5', weighted(mask & (rank > (length-5)[:, None])))
        variance = np.maximum((hc*hc*mask[:, None, :]).sum(2)/length.clip(1)[:, None]-mean*mean, 0)
        add('hist_std', np.sqrt(variance))
        pooled = h.sum(1)/length.clip(1)[:, None]
        coherence = np.linalg.norm(pooled, axis=1)
        add('pooled_cos', np.einsum('bik,bk->bi', v, pooled)/coherence.clip(1e-6)[:, None], True)
        add('coherence', coherence)
        ts = data['timestamps'][ix]
        past = ts[:, None]//60-np.rint(age*60).astype(np.int64)
        same_bucket = ((past%1440)//360 == (ts%86400//21600)[:, None]) & mask
        same_weekend = ((((past//1440+3)%7)>=5) == (data['context'][ix, 2]>0)[:, None]) & mask
        add('same_bucket', weighted(same_bucket))
        add('same_weekend', weighted(same_weekend))
        disagreement = np.abs((av[ids]*au[users, None, :]).sum(2)-(bv[ids]*bu[users, None, :]).sum(2))
        add('cos_disagreement', disagreement)
        hca = np.einsum('bik,bjk->bij', av[ids], av[hist])*mask[:, None, :]
        hcb = np.einsum('bik,bjk->bij', bv[ids], bv[hist])*mask[:, None, :]
        add('hist_disagreement', np.abs((hca-hcb).sum(2))/length.clip(1)[:, None])
        pieces.append(np.stack(values, axis=2).astype(np.float32))
    result = np.concatenate(pieces).reshape(-1, len(names))
    assert np.isfinite(result).all()
    return result, names
