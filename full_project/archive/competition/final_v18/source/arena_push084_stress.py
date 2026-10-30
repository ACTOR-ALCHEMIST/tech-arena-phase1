"""Fixed missing-history stress diagnostics after all weights have been frozen."""
import json
import hashlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from arena_ood import ROOT, arrays, source, corrupt, write_json, sha
from arena_push084 import OUT


def run():
    from ctr_data import load_cache
    from arena_features import complete_features
    from arena_ood_features import extension_features
    from arena_ood_audit import baseline_predict, neural
    from arena_finalize import blend
    from arena_robust_mf import load_model, predict, refresh_model
    cutoff = '2025-01-26'
    torch.set_num_threads(2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    base, info, labels, _, _ = arrays(cutoff)
    _, data = load_cache(ROOT/'data/cache_v1', 'train')
    indices = labels['validation_indices']
    chosen = np.sort(np.random.default_rng(8971).choice(len(indices), min(12000, len(indices)), replace=False))
    ix = indices[chosen]
    raw = {k: np.array(v[ix]) for k, v in data.items() if k != 'labels'}
    local = np.arange(len(ix))
    y = labels['valid'].reshape(-1, 10)[chosen]
    selection = json.loads((OUT/'selection.json').read_text())
    incumbent = np.load(ROOT/'competition/final_v5/validation_26.npz')['predictions'][chosen]
    expected = np.load(OUT/'selected_validation_26.npz')['predictions'][chosen]
    native_path = ROOT/'competition/robust_mf/runs/causal_frozen'/cutoff
    native = load_model(native_path, device)
    result = dict(cutoff=cutoff, events=len(ix), indices_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
                  selection_sha256=sha(OUT/'selection.json'), stress={},
                  interpretation='Artificial missing-input stress only. Original targets and ten candidates retained. No weights or epochs selected on this diagnostic.')
    checks = dict(identity=dict(mode='thin', drop_probability=0), thin15=dict(mode='thin', drop_probability=.15),
                  thin60=dict(mode='thin', drop_probability=.60), cold_user=dict(mode='cold_user'))
    for name, spec in checks.items():
        inputs = corrupt(raw, spec, seed=198719)
        x, names = complete_features(inputs, local, cutoff, base['feature_config'])
        assert names == info['feature_names']
        extras = {}
        def extended(mode):
            key = str(mode)
            if key not in extras:
                extra, _ = extension_features(inputs, local, cutoff, base['feature_config'], mode)
                extras[key] = np.concatenate([x, extra], 1)
            return extras[key]
        v3 = baseline_predict(cutoff, x, names)
        v4 = blend([v3, neural(ROOT/'competition/ood/runs/rich_wide_thin15'/cutoff, extended(True))], [.5, .5], 'logit')
        old = blend([v4, predict(native, inputs, local, device)], [.9, .1], 'logit')
        parts, weights = [old], [selection['selected']['weights'][0]]
        for model_name, weight in zip(selection['sources'][1:], selection['selected']['weights'][1:]):
            if weight <= 0:
                continue
            if model_name == 'refreshed_mf':
                p = predict(refresh_model(native, cutoff, device), inputs, local, device)
            else:
                path = ROOT/'competition/ood/runs'/model_name/cutoff
                r = json.loads((path/'result.json').read_text())
                aug = r['augmentation_config']
                mode = aug.get('feature_mode', aug.get('expanded', False))
                values = extended(mode) if mode else x
                if aug.get('origin_days'):
                    from arena_origins import delay
                    values = np.concatenate([values, delay(inputs, local, cutoff)], 1)
                p = neural(path, values)
            parts.append(p)
            weights.append(weight)
        new = blend(parts, weights, 'logit')
        if name == 'identity':
            old_error, new_error = float(np.max(np.abs(old-incumbent))), float(np.max(np.abs(new-expected)))
            assert old_error < 2e-5 and new_error < 2e-5, (old_error, new_error)
            result['identity_replay_max_error'] = dict(incumbent=old_error, selected=new_error)
        a, b = roc_auc_score(y.ravel(), old.ravel()), roc_auc_score(y.ravel(), new.ravel())
        result['stress'][name] = dict(old_auc=float(a), new_auc=float(b), delta=float(b-a))
        write_json(OUT/'stress.json', result)
        print(json.dumps(dict(stage='missing_history_stress', condition=name, **result['stress'][name])), flush=True)


if __name__ == '__main__':
    run()
