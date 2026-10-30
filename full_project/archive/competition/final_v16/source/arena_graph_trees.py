"""CPU boosted-tree controls using the new graph and repeat-memory features."""
import fcntl
import json
import time
from pathlib import Path
import numpy as np
import arena_core  # Adds the workspace's existing CatBoost installation.
from catboost import CatBoostClassifier
from arena_ood import ROOT, OUT, guard, sha, write_json


def matrix(values, names, context=False):
    x = np.ascontiguousarray(values[:], dtype=np.float32)
    if not context:
        return x, names
    # Only a compact, fixed semantic subset; no candidate positions or IDs.
    wanted = ['prior_logit', 'user_bias', 'item_bias', 'repeat_any', 'repeat_count',
              'cf_cos_sum', 'cf_cos_max', 'cf_recent_3d', 'als8_known_dot', 'als16_known_dot']
    wanted += [n for n in names if (n.startswith('sgns') and (n.endswith('_cos') or n.endswith('_dot')))
               or n.endswith('posterior_logit_delta') or n.endswith('ever_clicked')]
    selected = [names.index(n) for n in dict.fromkeys(wanted) if n in names]
    a = x.reshape(-1, 10, len(names))[:, :, selected]
    extra, extra_names = [], []
    for suffix, stat in [('slate_mean', a.mean(1)), ('slate_max', a.max(1)), ('slate_std', a.std(1))]:
        extra.append(np.repeat(stat, 10, axis=0))
        extra_names += [names[i]+'_'+suffix for i in selected]
    return np.ascontiguousarray(np.concatenate([x]+extra, 1)), names+extra_names


def predict(path, x):
    if (Path(path)/'model.txt').exists():
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(Path(path)/'model.txt'))
        if (Path(path)/'base_model.pt').exists():
            z = model.predict(np.ascontiguousarray(x), raw_score=True, num_threads=2)+neural_logit(Path(path)/'base_model.pt', x)
            from scipy.special import expit
            return expit(z).reshape(-1, 10)
        return model.predict(np.ascontiguousarray(x), num_threads=2).reshape(-1, 10)
    model = CatBoostClassifier()
    model.load_model(str(Path(path)/'model.cbm'))
    return model.predict_proba(np.ascontiguousarray(x))[:, 1].reshape(-1, 10)


def neural_logit(checkpoint, x):
    import torch
    from arena_push_model import build_model
    from arena_tabular import predict as neural_predict
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = saved['state_dict']
    model = build_model(saved['width'], state['mean'].numpy(), state['scale'].numpy(), saved['config'])
    model.load_state_dict(state)
    torch.set_num_threads(2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    p = neural_predict(model, x[:, :saved['width']], device).ravel().astype(np.float64).clip(1e-7, 1-1e-7)
    del model
    if device.type == 'mps':
        torch.mps.empty_cache()
    return np.log(p)-np.log1p(-p)


def fit_model(config, x, y, v, vy, fixed_epochs=None):
    params = dict(iterations=config.get('iterations', 1400) if fixed_epochs is None else fixed_epochs,
                  depth=config.get('depth', 6), learning_rate=.04, l2_leaf_reg=config.get('l2', 20),
                  loss_function='Logloss', eval_metric='AUC', thread_count=4, random_seed=2026,
                  border_count=128, allow_writing_files=False, verbose=100)
    model = CatBoostClassifier(**params)
    if fixed_epochs is None:
        model.fit(x, y, eval_set=(v, vy), early_stopping_rounds=100, use_best_model=True)
    else:
        assert fixed_epochs > 0
        model.fit(x, y)
    return model


def train(name, cutoff, fixed_epochs, overrides):
    from arena_ood import arrays
    from arena_core import evaluate
    from arena_validation import importance_weights, weighted_auc
    config, before = overrides['graph_tree'], guard()
    out = OUT/'runs'/name/cutoff
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'result.json').exists():
            result = json.loads((out/'result.json').read_text())
            assert result['augmentation_config'] == overrides
            return result
        base, info, labels, x, v = arrays(cutoff, 'spectral_multi_pair')
        x, names = matrix(x, info['feature_names'], config.get('slate_context'))
        v, vn = matrix(v, info['feature_names'], config.get('slate_context'))
        assert names == vn
        started = time.monotonic()
        model = fit_model(config, x, labels['train'].ravel(), v, labels['valid'].ravel(), fixed_epochs)
        model.save_model(str(out/'model.cbm'))
        p = model.predict_proba(v)[:, 1].reshape(-1, 10)
        np.savez_compressed(out/'validation.npz', event_indices=labels['validation_indices'], labels=labels['valid'].reshape(-1, 10), predictions=p)
        iw = importance_weights(cutoff, labels['validation_indices'])[0]
        result = dict(name=name, cutoff=cutoff, method='graph_catboost', config=config, augmentation_config=overrides,
                      best_epoch=model.tree_count_, **evaluate(labels['valid'], p),
                      weighted_auc=weighted_auc(labels['valid'], p, iw), feature_config=base['feature_config'],
                      feature_dataset=base['feature_dataset'], feature_names=names,
                      parameters=int(model.get_leaf_values().size), seconds=time.monotonic()-started,
                      input_layout='C-contiguous float32')
        # This fit has no dependency on incumbent predictions or selection.
        # Another authorized campaign may advance only the pointer meanwhile.
        assert all(sha(ROOT/p) == digest for p, digest in before.items()
                   if p != 'competition/CURRENT_BEST.json')
        write_json(out/'result.json', result)
        print(json.dumps(result), flush=True)
        return result


def fit_final(name, final):
    from arena_ood_export import final_inputs
    r = json.loads((OUT/'runs'/name/'2025-01-26/result.json').read_text())
    out = final/'models'/name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'recipe.json').exists():
        return out
    base, raw, evaluation, indices, x, v, names = final_inputs('spectral_multi_pair')
    original_names = names
    x, names = matrix(x, original_names, r['config'].get('slate_context'))
    v, vn = matrix(v, original_names, r['config'].get('slate_context'))
    assert names == vn
    model = fit_model(r['config'], x, raw['labels'][indices].ravel(), v, None, r['best_epoch'])
    model.save_model(str(out/'model.cbm'))
    p = model.predict_proba(v)[:, 1].reshape(-1, 10)
    np.save(out/'evaluation_predictions.npy', p)
    write_json(out/'recipe.json', dict(name=name, graph_tree=True, augmentation_config=r['augmentation_config'],
               feature_config=base['feature_config'], feature_names=names, model_config=r['config'],
               training_events=len(indices), fixed_epochs=r['best_epoch'], model_sha256=sha(out/'model.cbm')))
    return out
