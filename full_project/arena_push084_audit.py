"""Frozen-recipe third time origin and event-level uncertainty diagnostics."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from arena_ood import ROOT, sha, write_json
from arena_push084 import OUT, command


def sorted_auc(labels, predictions):
    y, p = labels.ravel(), predictions.ravel()
    order = np.argsort(p, kind='stable')
    ys, ps = y[order].astype(bool), p[order]
    starts = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1])+1]
    def calc(event_weights):
        w = np.repeat(event_weights, 10)[order]
        pos = np.add.reduceat(w*ys, starts)
        neg = np.add.reduceat(w*~ys, starts)
        return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))
    return calc


def compare(old, new, cutoff, repetitions=200):
    from ctr_data import load_cache
    from run import timestamp
    for key in ['event_indices', 'labels']:
        np.testing.assert_array_equal(old[key], new[key])
    labels, ix = old['labels'], old['event_indices']
    a, b = old['predictions'], new['predictions']
    f, g = sorted_auc(labels, a), sorted_auc(labels, b)
    ones = np.ones(len(labels))
    old_auc, new_auc = roc_auc_score(labels.ravel(), a.ravel()), roc_auc_score(labels.ravel(), b.ravel())
    assert abs(f(ones)-old_auc) < 1e-12 and abs(g(ones)-new_auc) < 1e-12
    rng = np.random.default_rng(8024)
    deltas = []
    for _ in range(repetitions):
        w = rng.poisson(1, len(labels))
        deltas.append(g(w)-f(w))
    meta, data = load_cache(ROOT/'data/cache_v1', 'train')
    known = np.zeros(meta['n_users'], bool)
    known[data['users'][data['timestamps'] < timestamp(cutoff)]] = True
    length = (data['history'][ix] != 0).sum(1)
    elapsed = (data['timestamps'][ix]-timestamp(cutoff))/3600
    cohorts = {}
    for name, mask in dict(known_user=known[data['users'][ix]], new_user=~known[data['users'][ix]],
                           empty_history=length == 0, short_history=(length > 0)&(length < 5),
                           history5_19=(length >= 5)&(length < 20), history20=length == 20,
                           delay0_24=elapsed < 24, delay24_48=(elapsed >= 24)&(elapsed < 48), delay48plus=elapsed >= 48,
                           weekday=data['context'][ix, 2] == 0, weekend=data['context'][ix, 2] > 0).items():
        y = labels[mask].ravel()
        if mask.any() and len(np.unique(y)) == 2:
            old_value, new_value = roc_auc_score(y, a[mask].ravel()), roc_auc_score(y, b[mask].ravel())
            cohorts[name] = dict(events=int(mask.sum()), old_auc=float(old_value), new_auc=float(new_value), delta=float(new_value-old_value))
    return dict(cutoff=cutoff, old_auc=float(old_auc), new_auc=float(new_auc), delta=float(new_auc-old_auc),
                event_poisson_bootstrap=dict(repetitions=repetitions, seed=8024,
                delta_percentile95=np.quantile(deltas, [.025, .975]).tolist(), positive_fraction=float(np.mean(np.array(deltas) > 0)),
                limitation='Fixed selected recipe only; does not correct for experiment selection bias.'), cohorts=cohorts)


def third():
    from arena_finalize import blend
    from arena_robust_mf import load_model, refresh_model, predict
    from ctr_data import load_cache
    import torch
    selection = json.loads((OUT/'selection.json').read_text())
    command(['arena_robust_mf.py', '--name', 'causal_frozen', '--cutoff', '2025-01-24', '--fixed-epochs', '11'], 'baseline_native_locked24')
    previous = np.load(ROOT/'competition/final_v4/validation_24.npz')
    native_path = ROOT/'competition/robust_mf/runs/causal_frozen/2025-01-24'
    native = np.load(native_path/'validation.npz')
    for k in ['event_indices', 'labels']:
        np.testing.assert_array_equal(previous[k], native[k])
    old = dict(event_indices=previous['event_indices'], labels=previous['labels'], predictions=blend([previous['predictions'], native['predictions']], [.9, .1], 'logit'))
    parts = [old['predictions']]
    weights = [selection['selected']['weights'][0]]
    for name, weight in zip(selection['sources'][1:], selection['selected']['weights'][1:]):
        if weight <= 0:
            continue
        if name == 'refreshed_mf':
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            torch.set_num_threads(2)
            _, data = load_cache(ROOT/'data/cache_v1', 'train')
            model = refresh_model(load_model(native_path, device), '2025-01-24', device)
            p = predict(model, data, old['event_indices'], device)
        else:
            primary = json.loads((ROOT/'competition/ood/runs'/name/'2025-01-26/result.json').read_text())
            command(['arena_ood.py', 'train', '--name', name, '--config', 'configs/'+name+'.json', '--cutoff', '2025-01-24',
                     '--fixed-epochs', str(primary['best_epoch'])], name+'_locked24')
            d = np.load(ROOT/'competition/ood/runs'/name/'2025-01-24/validation.npz')
            for k in ['event_indices', 'labels']:
                np.testing.assert_array_equal(d[k], old[k])
            p = d['predictions']
        parts.append(p)
        weights.append(weight)
    new = dict(old, predictions=blend(parts, weights, selection['selected']['kind']))
    np.savez_compressed(OUT/'incumbent_validation_24.npz', **old)
    np.savez_compressed(OUT/'selected_validation_24.npz', **new)
    report = compare(old, new, '2025-01-24')
    report.update(protocol='Weights and epochs frozen on Jan26/Jan25 before this third origin; no search on Jan24.',
                  selection_sha256=sha(OUT/'selection.json'))
    write_json(OUT/'audit_24.json', report)
    print(json.dumps(report, indent=2), flush=True)


def audits():
    selection = json.loads((OUT/'selection.json').read_text())
    pointer = json.loads((OUT/'incumbent_pointer.json').read_text())
    folder = (ROOT/pointer['csv']).parent
    for day in ['26', '25']:
        result = compare(np.load(folder/('validation_'+day+'.npz')),
                         np.load(OUT/('selected_validation_'+day+'.npz')), '2025-01-'+day)
        result['selection_sha256'] = sha(OUT/'selection.json')
        write_json(OUT/('audit_'+day+'.json'), result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    third() if '--third' in sys.argv else audits()
