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
    if mode == 'fused_sphere':
        from arena_fused_sphere import extra_features as fused
        return fused(data,indices,cutoff,config)
    if mode in ['fused_spectral','fused_fresh','fused_stack']:
        inner={'fused_spectral':'spectral_multi','fused_fresh':'fresh_spectral','fused_stack':'long_stack'}[mode]
        a,na=extension_features(data,indices,cutoff,config,inner)
        b,nb=extension_features(data,indices,cutoff,config,'fused_sphere')
        return np.concatenate([a,b],1),na+nb
    if mode == 'sphere_long':
        from arena_spherical_long import extra_features as long_sphere
        return long_sphere(data,indices,cutoff,config)
    if mode in ['long_spectral','long_fresh','long_stack']:
        inner={'long_spectral':'spectral_multi','long_fresh':'fresh_spectral','long_stack':'spherical_fresh'}[mode]
        a,na=extension_features(data,indices,cutoff,config,inner)
        b,nb=extension_features(data,indices,cutoff,config,'sphere_long')
        return np.concatenate([a,b],1),na+nb
    if mode == 'aligned':
        from arena_aligned_repr import extra_features as aligned
        return aligned(data, indices, cutoff, config)
    if mode in ['aligned_spectral', 'aligned_sphere', 'aligned_fresh']:
        inner = {'aligned_spectral':'spectral_multi', 'aligned_sphere':'spherical_spectral', 'aligned_fresh':'spherical_fresh'}[mode]
        a, na = extension_features(data, indices, cutoff, config, inner)
        b, nb = extension_features(data, indices, cutoff, config, 'aligned')
        return np.concatenate([a, b], 1), na+nb
    if mode == 'spherical':
        from arena_spherical_repr import extra_features as spherical
        return spherical(data, indices, cutoff, config)
    if mode in ['spherical_spectral', 'spherical_fresh']:
        inner = 'fresh_spectral' if mode == 'spherical_fresh' else 'spectral_multi'
        a, na = extension_features(data, indices, cutoff, config, inner)
        b, nb = extension_features(data, indices, cutoff, config, 'spherical')
        return np.concatenate([a, b], 1), na+nb
    if mode == 'ppmi':
        from arena_ppmi import extra_features as ppmi
        return ppmi(data, indices, cutoff, config)
    if mode in ['ppmi_spectral', 'ppmi_fresh']:
        inner = 'fresh_spectral' if mode == 'ppmi_fresh' else 'spectral_multi'
        a, na = extension_features(data, indices, cutoff, config, inner)
        b, nb = extension_features(data, indices, cutoff, config, 'ppmi')
        return np.concatenate([a, b], 1), na+nb
    if mode == 'online_events':
        from arena_online_events import extra_features as online
        return online(data, indices, cutoff, config)
    if mode in ['online_spectral', 'online_fresh']:
        inner = 'fresh_spectral' if mode == 'online_fresh' else 'spectral_multi'
        a, na = extension_features(data, indices, cutoff, config, inner)
        b, nb = extension_features(data, indices, cutoff, config, 'online_events')
        return np.concatenate([a, b], 1), na+nb
    if mode == 'fresh_mf':
        from arena_supervised_repr import extra_features as supervised
        return supervised(data, indices, cutoff, config)
    if mode == 'fresh_spectral':
        a, na = extension_features(data, indices, cutoff, config, 'spectral_multi')
        b, nb = extension_features(data, indices, cutoff, config, 'fresh_mf')
        return np.concatenate([a, b], 1), na+nb
    if mode == 'novel_choice':
        from arena_novel_choice import extra_features as novel
        return novel(data, indices, cutoff, config)
    if mode == 'novel_spectral':
        a, na = extension_features(data, indices, cutoff, config, 'spectral')
        b, nb = extension_features(data, indices, cutoff, config, 'novel_choice')
        return np.concatenate([a, b], 1), na+nb
    if mode == 'spectral_multi_pair':
        a, na = extension_features(data, indices, cutoff, config, 'spectral_multi')
        b, nb = extension_features(data, indices, cutoff, config, 'pair_memory')
        return np.concatenate([a, b], 1), na+nb
    if mode == 'pair_memory':
        from arena_pair_memory import extra_features as memory
        return memory(data, indices, cutoff, config)
    if mode in ['pair_coclick', 'pair_spectral']:
        inner = 'coclick' if mode == 'pair_coclick' else 'spectral'
        a, na = extension_features(data, indices, cutoff, config, inner)
        b, nb = extension_features(data, indices, cutoff, config, 'pair_memory')
        return np.concatenate([a, b], 1), na+nb
    if mode in ['residual8', 'residual24', 'frozen_graph16']:
        from arena_residual_adapter import extra_features as adapter
        return adapter(data, indices, cutoff, config, mode)
    if mode == 'coclick_rich':
        a, na = extra_features(data, indices, cutoff, config)
        b, nb = extension_features(data, indices, cutoff, config, 'coclick')
        return np.concatenate([a, b], axis=1), na+nb
    if mode == 'coclick':
        from arena_coclick import extra_features as coclick
        return coclick(data, indices, cutoff, config)
    if mode == 'exposure':
        from arena_exposure_repr import extra_features as exposure
        return exposure(data, indices, cutoff, config)
    if mode == 'spectral':
        from arena_spectral import extra_features as spectral
        return spectral(data, indices, cutoff, config)
    if mode == 'spectral64':
        from arena_spectral import extra_features as spectral
        return spectral(data, indices, cutoff, config, 64)
    if mode == 'spectral_multi':
        a, na = extension_features(data, indices, cutoff, config, 'spectral')
        b, nb = extension_features(data, indices, cutoff, config, 'spectral64')
        return np.concatenate([a, b], 1), na+nb
    if mode in ['softmax30', 'softmax90']:
        from arena_softmax_repr import extra_features as conditional
        return conditional(data, indices, cutoff, config, int(mode[7:]))
    if mode == 'joint_graph':
        from arena_joint_graph import extra_features as joint
        return joint(data, indices, cutoff, config)
    if mode in ['bias80', 'bias240']:
        from arena_bias_repr import extra_features as biased
        return biased(data, indices, cutoff, config, int(mode[4:]))
    if mode == 'weekend':
        from arena_weekend import extra_features as weekday_weekend
        return weekday_weekend(data, indices, cutoff, config)
    if mode == 'context':
        from arena_ood_context import extra_features as context
        return context(data, indices, cutoff, config)
    if mode == 'consensus':
        from arena_ood_consensus import extra_features as consensus
        return consensus(data, indices, cutoff, config)
    return extra_features(data, indices, cutoff, config)


def dataset(cutoff, base, labels, train, valid, info, mode=True):
    if mode in ['fused_spectral','fused_fresh','fused_stack']:
        inner={'fused_spectral':'spectral_multi','fused_fresh':'fresh_spectral','fused_stack':'long_stack'}[mode]
        base,info,labels,train,valid=dataset(cutoff,base,labels,train,valid,info,inner)
        return dataset(cutoff,base,labels,train,valid,info,'fused_sphere')
    if mode in ['long_spectral','long_fresh','long_stack']:
        inner={'long_spectral':'spectral_multi','long_fresh':'fresh_spectral','long_stack':'spherical_fresh'}[mode]
        base,info,labels,train,valid=dataset(cutoff,base,labels,train,valid,info,inner)
        return dataset(cutoff,base,labels,train,valid,info,'sphere_long')
    if mode in ['aligned_spectral', 'aligned_sphere', 'aligned_fresh']:
        inner = {'aligned_spectral':'spectral_multi', 'aligned_sphere':'spherical_spectral', 'aligned_fresh':'spherical_fresh'}[mode]
        base, info, labels, train, valid = dataset(cutoff, base, labels, train, valid, info, inner)
        return dataset(cutoff, base, labels, train, valid, info, 'aligned')
    if mode in ['spherical_spectral', 'spherical_fresh']:
        inner = 'fresh_spectral' if mode == 'spherical_fresh' else 'spectral_multi'
        base, info, labels, train, valid = dataset(cutoff, base, labels, train, valid, info, inner)
        return dataset(cutoff, base, labels, train, valid, info, 'spherical')
    if mode in ['ppmi_spectral', 'ppmi_fresh']:
        inner = 'fresh_spectral' if mode == 'ppmi_fresh' else 'spectral_multi'
        base, info, labels, train, valid = dataset(cutoff, base, labels, train, valid, info, inner)
        return dataset(cutoff, base, labels, train, valid, info, 'ppmi')
    if mode in ['online_spectral', 'online_fresh']:
        inner = 'fresh_spectral' if mode == 'online_fresh' else 'spectral_multi'
        base, info, labels, train, valid = dataset(cutoff, base, labels, train, valid, info, inner)
        return dataset(cutoff, base, labels, train, valid, info, 'online_events')
    if mode == 'fresh_spectral':
        base, info, labels, train, valid = dataset(cutoff, base, labels, train, valid, info, 'spectral_multi')
        return dataset(cutoff, base, labels, train, valid, info, 'fresh_mf')
    if mode == 'novel_spectral':
        base, info, labels, train, valid = dataset(cutoff, base, labels, train, valid, info, 'spectral')
        return dataset(cutoff, base, labels, train, valid, info, 'novel_choice')
    if mode == 'spectral_multi_pair':
        base, info, labels, train, valid = dataset(cutoff, base, labels, train, valid, info, 'spectral_multi')
        return dataset(cutoff, base, labels, train, valid, info, 'pair_memory')
    if mode in ['pair_coclick', 'pair_spectral']:
        inner = 'coclick' if mode == 'pair_coclick' else 'spectral'
        base, info, labels, train, valid = dataset(cutoff, base, labels, train, valid, info, inner)
        return dataset(cutoff, base, labels, train, valid, info, 'pair_memory')
    if mode == 'spectral_multi':
        base, info, labels, train, valid = dataset(cutoff, base, labels, train, valid, info, 'spectral')
        return dataset(cutoff, base, labels, train, valid, info, 'spectral64')
    if mode == 'coclick_rich':
        base, info, labels, train, valid = dataset(cutoff, base, labels, train, valid, info, True)
        return dataset(cutoff, base, labels, train, valid, info, 'coclick')
    from arena_ood import OUT, ROOT, write_json
    from ctr_data import load_cache
    key = hashlib.sha256(json.dumps(base['feature_config'], sort_keys=True).encode()).hexdigest()[:12]
    folder = OUT/'features'/(cutoff+'_'+key+('_'+mode if isinstance(mode, str) else ''))
    folder.mkdir(parents=True, exist_ok=True)
    _, data = load_cache(ROOT/'data/cache_v1', 'train')
    lazy_train = None
    for phase, indices, frame in [('train', labels['train_indices'], base['feature_config']['train_start']),
                                  ('valid', labels['validation_indices'], cutoff)]:
        if phase == 'train' and mode == 'aligned':
            from arena_aligned_repr import lazy_features
            lazy_train, names = lazy_features(data, indices, frame)
            write_json(folder/(phase+'.json'), {'names': names, 'events': len(indices), 'frame': frame,
                       'indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(), 'storage': 'lazy_float32_rows'})
            continue
        if not (folder/(phase+'.json')).exists() or not (folder/(phase+'.npy')).exists():
            config = dict(base['feature_config'], online_cap=cutoff)
            x, names = extension_features(data, indices, frame, config, mode)
            np.save(folder/(phase+'.npy'), x)
            write_json(folder/(phase+'.json'), {'names': names, 'events': len(indices), 'frame': frame,
                       'indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
                       **({'online_cap': cutoff} if mode == 'online_events' else {})})
            print(json.dumps({'stage': 'expanded_history_features', 'phase': phase, 'cutoff': cutoff, 'shape': x.shape}), flush=True)
    names = json.loads((folder/'train.json').read_text())['names']
    assert names == json.loads((folder/'valid.json').read_text())['names']
    info = dict(info, feature_names=info['feature_names']+names)
    training_extra = lazy_train if lazy_train is not None else np.load(folder/'train.npy', mmap_mode='r')
    return base, info, labels, FeatureMatrix(train, training_extra), FeatureMatrix(valid, np.load(folder/'valid.npy', mmap_mode='r'))
