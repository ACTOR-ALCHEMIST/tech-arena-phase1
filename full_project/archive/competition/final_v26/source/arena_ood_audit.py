"""Natural-cohort and matched corruption tests, separate from clean selection."""
import argparse
import json
import numpy as np
import torch
from arena_ood import ROOT, OUT, arrays, corrupt, write_json, sha


def neural(path, x):
    from arena_tabular import predict
    from arena_push_model import build_model
    saved = torch.load(path/'model.pt', map_location='cpu', weights_only=True)
    state = saved['state_dict']
    model = build_model(saved['width'], state['mean'].numpy(), state['scale'].numpy(), saved['config'])
    model.load_state_dict(state)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    return predict(model, x, device)


def baseline_predict(cutoff, x, names):
    from arena_finalize import blend
    from catboost import CatBoostClassifier
    from arena_event import aggregate, event_probability
    sources = ['event_cat_sgns8e640_context24_d6', 'mlp_event_cat_sgns8_s2026_context24_d6',
               'v2_multiview_wide256_s2026', 'v2_multiview_cross256_s2026', 'v3_cross512_s2027']
    parts = []
    for name in sources:
        suffix = '' if cutoff.endswith('26') else '_locked'+cutoff[-2:]
        path = ROOT/'competition/experiments'/(name+suffix)
        if name == 'v3_cross512_s2027' and cutoff.endswith('24'):
            path = OUT/'runs'/'control24'/cutoff
        r = json.loads((path/'result.json').read_text())
        columns = json.loads((ROOT/r['feature_dataset']/'metadata.json').read_text())['feature_names']
        # Advanced column indexing returns a Fortran-layout array. Preserve the
        # original C reduction order in the legacy tree gate's mean/std inputs;
        # tiny rounding differences around tree thresholds change predictions.
        values = np.ascontiguousarray(x[:, [names.index(c) for c in columns]])
        if r['method'] == 'catboost_event':
            point = CatBoostClassifier()
            point.load_model(str(path/'model.cbm'))
            gate = CatBoostClassifier()
            gate.load_model(str(path/'event_gate.cbm'))
            p = event_probability(point.predict_proba(values, thread_count=2)[:, 1].reshape(-1, 10),
                                  gate.predict_proba(aggregate(values), thread_count=2)[:, 1],
                                  r['temperature'], r['event_mix'])
        else:
            p = neural(path, values)
        parts.append(p)
    # Reproduce the nested composition to preserve floating point semantics.
    return blend([blend(parts[:4], [.25]*4, 'logit'), parts[4]], [.9, .1], 'logit')


def cohort_metrics(data, indices, cutoff, labels, predictions):
    from sklearn.metrics import roc_auc_score
    from arena_core import snapshot
    stats = np.load(snapshot(cutoff)/'stats.npz')
    known = stats['known_users'][data['users'][indices]]
    length = np.count_nonzero(data['history'][indices], axis=1)
    groups = {'all': np.ones(len(indices), bool), 'unseen_user': ~known, 'known_user': known,
              'history_0': length == 0, 'history_1_4': (length >= 1) & (length < 5),
              'history_5_9': (length >= 5) & (length < 10),
              'history_10_19': (length >= 10) & (length < 20), 'history_20': length == 20}
    result = {}
    for name, mask in groups.items():
        y, p = labels[mask].ravel(), predictions[mask].ravel()
        result[name] = {'events': int(mask.sum()), 'positive_rows': int(y.sum()),
                        'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}
    return result


def run(names, cutoff='2025-01-26'):
    from ctr_data import load_cache
    from arena_features import complete_features
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(2)
    base, info, labels, _, valid = arrays(cutoff)
    _, data = load_cache(ROOT/'data/cache_v1', 'train')
    indices = labels['validation_indices']
    y = labels['valid'].reshape(-1, 10)
    result = {'cutoff': cutoff, 'cohorts': {}, 'stress': {}}
    frozen_path = ROOT/'competition/v3'/('selected_validation_'+cutoff[-2:]+'.npz')
    if cutoff.endswith('24'):
        frozen_path = OUT/'frozen_validation_24.npz'
        if not frozen_path.exists():
            p = baseline_predict(cutoff, valid, info['feature_names'])
            np.savez_compressed(frozen_path, event_indices=indices, labels=y, predictions=p)
    frozen = np.load(frozen_path)
    np.testing.assert_array_equal(frozen['event_indices'], indices)
    result['cohorts']['frozen_v3'] = cohort_metrics(data, indices, cutoff, y, frozen['predictions'])
    control = ROOT/'competition/experiments'/('v3_cross512_s2027'+('' if cutoff.endswith('26') else '_locked'+cutoff[-2:]))
    if cutoff.endswith('24'):
        control = OUT/'runs'/'control24'/cutoff
    result['cohorts']['clean_cross_control'] = cohort_metrics(data, indices, cutoff, y, np.load(control/'validation.npz')['predictions'])
    for name in names:
        path = OUT/'runs'/name/cutoff
        d = np.load(path/'validation.npz')
        np.testing.assert_array_equal(d['event_indices'], indices)
        result['cohorts'][name] = cohort_metrics(data, indices, cutoff, y, d['predictions'])
    selection = json.loads((OUT/'selection.json').read_text())
    active = [(n, w) for n, w in zip(selection['sources'][1:], selection['selected']['weights'][1:]) if w > 0]
    if all(n in names for n, _ in active):
        from arena_finalize import blend
        parts = [frozen['predictions']]+[np.load(OUT/'runs'/n/cutoff/'validation.npz')['predictions'] for n, _ in active]
        weights = [selection['selected']['weights'][0]]+[w for _, w in active]
        p = blend(parts, weights, 'logit')
        result['cohorts']['selected_ensemble'] = cohort_metrics(data, indices, cutoff, y, p)
        np.savez_compressed(OUT/('audited_validation_'+cutoff[-2:]+'.npz'), event_indices=indices, labels=y, predictions=p)
    # Same fixed events and corruptions for every scorer; never used for epochs.
    chosen = np.sort(np.random.default_rng(8971).choice(len(indices), min(12000, len(indices)), replace=False))
    inputs = {k: np.array(v[indices[chosen]]) for k, v in data.items() if k != 'labels'}
    checks = {'identity': {'mode': 'thin', 'drop_probability': 0},
              'thin30': {'mode': 'thin', 'drop_probability': .30},
              'thin60': {'mode': 'thin', 'drop_probability': .60},
              'recent5': {'mode': 'truncate', 'limits': [5]},
              'cold_user': {'mode': 'cold_user'}}
    for name, spec in checks.items():
        perturbed = corrupt(inputs, spec, seed=198719)
        x, columns = complete_features(perturbed, np.arange(len(chosen)), cutoff, base['feature_config'])
        assert columns == info['feature_names'] and np.isfinite(x).all()
        if name == 'identity':
            clean = np.asarray(valid).reshape(-1, 10, valid.shape[1])[chosen].reshape(-1, valid.shape[1])
            error = float(np.max(np.abs(x-clean)))
            assert error < 2e-6, error
            result['identity_features_max_error'] = error
        predictions = {'frozen_v3': baseline_predict(cutoff, x, columns), 'clean_cross_control': neural(control, x)}
        expanded = {}
        for model_name in names:
            path = OUT/'runs'/model_name/cutoff
            r = json.loads((path/'result.json').read_text())
            extension = r['augmentation_config'].get('feature_mode', r['augmentation_config'].get('expanded', False))
            if extension:
                if extension not in expanded:
                    from arena_ood_features import extension_features
                    extra, _ = extension_features(perturbed, np.arange(len(chosen)), cutoff, base['feature_config'], extension)
                    expanded[extension] = np.concatenate([x, extra], axis=1)
                predictions[model_name] = neural(path, expanded[extension])
            else:
                predictions[model_name] = neural(path, x)
        if name == 'identity':
            error = float(np.max(np.abs(predictions['frozen_v3']-frozen['predictions'][chosen])))
            assert error < 2e-5, error
            result['baseline_replay_max_error'] = error
        result['stress'][name] = {key: float(roc_auc_score(y[chosen].ravel(), p.ravel())) for key, p in predictions.items()}
        if all(n in predictions for n, _ in active):
            from arena_finalize import blend
            p = blend([predictions['frozen_v3']]+[predictions[n] for n, _ in active],
                      [selection['selected']['weights'][0]]+[w for _, w in active], 'logit')
            result['stress'][name]['selected_ensemble'] = float(roc_auc_score(y[chosen].ravel(), p.ravel()))
        print(json.dumps({'cutoff': cutoff, 'stress': name, 'auc': result['stress'][name]}), flush=True)
        write_json(OUT/('audit_'+cutoff[-2:]+'.json'), result)
    result['stress_events'] = len(chosen)
    result['stress_indices_sha256'] = __import__('hashlib').sha256(indices[chosen].tobytes()).hexdigest()
    result['interpretation'] = 'Artificial missing-history robustness and natural cohort AUC; not a guarantee of real future OOD or leaderboard AUC.'
    write_json(OUT/('audit_'+cutoff[-2:]+'.json'), result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--cutoff', default='2025-01-26')
    a = p.parse_args()
    run(a.names, a.cutoff)
