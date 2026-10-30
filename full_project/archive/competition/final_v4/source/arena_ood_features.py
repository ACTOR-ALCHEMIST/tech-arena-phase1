"""Extra history/context matching features in existing past-only SGNS frames."""
import hashlib
import json
from pathlib import Path
import numpy as np


class FeatureMatrix:
    """Column concatenation of mmap arrays, materializing only requested rows."""
    def __init__(self, *parts):
        assert len({len(x) for x in parts}) == 1
        self.parts = parts
        self.shape = (len(parts[0]), sum(x.shape[1] for x in parts))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return np.concatenate([np.asarray(x[index]) for x in self.parts], axis=-1)


def extra_features(data, indices, cutoff, config):
    from arena_sgns_features import embedding_model
    specs = [(f, config.get('sgns_recipes', {}).get(str(f)),
              config.get('sgns_epochs', {}).get(str(f))) for f in config['sgns_factors']]
    specs += [(x['factors'], x['recipe'], None) for x in config.get('sgns_extra', [])]
    names, pieces = [], []
    for factors, recipe, epochs in specs:
        model = embedding_model(cutoff, factors, epochs=epochs, recipe=recipe)
        seed = (recipe or {}).get('seed', 42)
        blocks = []
        for start in range(0, len(indices), 2048):
            ix = indices[start:start+2048]
            items = np.array(data['items'][ix])
            hist = np.array(data['history'][ix])
            age = np.array(data['age_hours'][ix])
            v = model['items'][items]
            v = v/np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-6)
            h = model['items'][hist]
            h = h/np.linalg.norm(h, axis=-1, keepdims=True).clip(1e-6)
            mask = hist != 0
            h = h*mask[..., None]
            cos = np.einsum('bik,bjk->bij', v, h)
            length = mask.sum(1)
            values, columns = [], []

            def add(name, value, relative=False):
                if value.ndim == 1:
                    value = np.broadcast_to(value[:, None], items.shape)
                values.append(value)
                columns.append(name)
                if relative:
                    values.append(value-value.mean(1, keepdims=True))
                    columns.append(name+'_relative')

            def weighted(weight):
                return (cos*weight[:, None, :]).sum(2)/weight.sum(1).clip(1e-6)[:, None]

            rank = np.cumsum(mask, axis=1)
            for count in [1, 3, 5]:
                last = mask & (rank > (length-count)[:, None])
                add('last'+str(count), weighted(last), relative=True)
            mean = weighted(mask)
            variance = np.maximum((cos*cos*mask[:, None, :]).sum(2)/length.clip(1)[:, None]-mean*mean, 0)
            add('hist_std', np.sqrt(variance))
            low = np.where(mask[:, None, :], cos, np.inf).min(2)
            add('hist_min', np.where(length[:, None] > 0, low, 0))
            sorted_cos = np.sort(np.where(mask[:, None, :], cos, -np.inf), axis=2)[:, :, -3:]
            top3 = np.where(np.isfinite(sorted_cos), sorted_cos, 0).sum(2)/np.minimum(length, 3).clip(1)[:, None]
            add('hist_top3', top3)
            for tau in [6, 24, 168]:
                add('decay'+str(tau)+'h', weighted(np.exp(-age/tau)*mask), relative=tau == 24)
            pooled = h.sum(1)/length.clip(1)[:, None]
            coherence = np.linalg.norm(pooled, axis=1)
            add('pooled_cos', np.einsum('bik,bk->bi', v, pooled)/coherence.clip(1e-6)[:, None])
            add('coherence', coherence)
            # Circular hour proximity is continuous across midnight.
            hours_apart = np.abs(np.mod(age+12, 24)-12)
            hour_weight = np.exp(-.5*(hours_apart/3)**2)*np.exp(-age/168)*mask
            add('near_hour3', weighted(hour_weight), relative=True)
            pair = np.einsum('bik,bjk->bij', v, v)
            add('slate_cos_mean', (pair.sum(2)-np.diagonal(pair, axis1=1, axis2=2))/9)
            pair[:, np.arange(10), np.arange(10)] = -np.inf
            add('slate_cos_max', pair.max(2))
            user = model['users'][data['users'][ix]]
            user = user/np.linalg.norm(user, axis=1, keepdims=True).clip(1e-6)
            add('user_history_cos', (user*pooled).sum(1)/coherence.clip(1e-6))
            blocks.append(np.stack(values, axis=2).astype(np.float32))
        pieces.append(np.concatenate(blocks).reshape(-1, len(columns)))
        names += [f'sgns{factors}_s{seed}_ood_{name}' for name in columns]
    result = np.concatenate(pieces, axis=1)
    assert result.shape == (len(indices)*10, len(names)) and np.isfinite(result).all()
    return result, names


def extension_features(data, indices, cutoff, config, mode=True):
    if mode == 'context':
        from arena_ood_context import extra_features as context
        return context(data, indices, cutoff, config)
    if mode == 'consensus':
        from arena_ood_consensus import extra_features as consensus
        return consensus(data, indices, cutoff, config)
    return extra_features(data, indices, cutoff, config)


def dataset(cutoff, base, labels, train, valid, info, mode=True):
    from arena_ood import OUT, ROOT, write_json
    from ctr_data import load_cache
    key = hashlib.sha256(json.dumps(base['feature_config'], sort_keys=True).encode()).hexdigest()[:12]
    folder = OUT/'features'/(cutoff+'_'+key+('_'+mode if isinstance(mode, str) else ''))
    folder.mkdir(parents=True, exist_ok=True)
    _, data = load_cache(ROOT/'data/cache_v1', 'train')
    for phase, indices, frame in [('train', labels['train_indices'], base['feature_config']['train_start']),
                                  ('valid', labels['validation_indices'], cutoff)]:
        if not (folder/(phase+'.json')).exists():
            x, names = extension_features(data, indices, frame, base['feature_config'], mode)
            np.save(folder/(phase+'.npy'), x)
            write_json(folder/(phase+'.json'), {'names': names, 'events': len(indices), 'frame': frame,
                       'indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest()})
            print(json.dumps({'stage': 'expanded_history_features', 'phase': phase, 'cutoff': cutoff, 'shape': x.shape}), flush=True)
    names = json.loads((folder/'train.json').read_text())['names']
    assert names == json.loads((folder/'valid.json').read_text())['names']
    info = dict(info, feature_names=info['feature_names']+names)
    return base, info, labels, FeatureMatrix(train, np.load(folder/'train.npy', mmap_mode='r')), FeatureMatrix(valid, np.load(folder/'valid.npy', mmap_mode='r'))
