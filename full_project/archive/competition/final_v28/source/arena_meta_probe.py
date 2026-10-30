"""Diagnostic forward calibration using earlier, out-of-time head predictions.

Fit only Jan24 calendar-day targets. Jan25/26 are forward diagnostics. This
cannot provide an untouched Jan24 audit, so it never exports or promotes a CSV.
An improvement would justify preparing a still-earlier rolling prediction fold.
"""
import json
import time
import numpy as np
import arena_core
import lightgbm as lgb
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from arena_ood import ROOT, sha, write_json

OUT = ROOT/'competition/meta_probe'


def inputs(day, parent, source_names, raw):
    from arena_core import snapshot
    d = dict(np.load(parent/('validation_'+day+'.npz')))
    ix, p = d['event_indices'], d['predictions'].astype(np.float64).clip(1e-7, 1-1e-7)
    z = logit(p)
    mass = p.sum(1).clip(1e-7, 1-1e-7)
    ctx = np.asarray(raw['context'][ix])
    h, items = raw['history'][ix], raw['items'][ix]
    repeat = (items[:, :, None] == h[:, None]) & (h[:, None] != 0)
    ages = raw['age_hours'][ix]
    stats = np.load(snapshot('2025-01-'+day)/'stats.npz')
    values, names = [], []

    def add(name, value):
        if value.ndim == 1:
            value = np.broadcast_to(value[:, None], p.shape)
        values.append(value)
        names.append(name)

    add('baseline_logit', z)
    add('event_mass_logit', logit(mass))
    add('conditional_log_probability', np.log(p)-np.log(p.sum(1))[:, None])
    add('event_max', p.max(1))
    add('event_std', p.std(1))
    add('event_entropy', -(p*np.log(p)).sum(1))
    for k, name in enumerate(['hour_sin', 'hour_cos', 'weekend', 'history_length', 'last_gap', 'history_span']):
        add(name, ctx[:, k])
    add('repeat_any', repeat.any(2))
    add('repeat_count_log', np.log1p(repeat.sum(2)))
    add('repeat72', (repeat*np.exp(-ages[:, None]/72)).sum(2))
    add('user_support', np.log1p(stats['ui'][raw['users'][ix]]))
    add('item_support', np.log1p(stats['ii'][items]))
    add('item_positive_support', np.log1p(stats['pop'][items]))
    for name in source_names:
        q = np.load(ROOT/'competition/ood/runs'/name/('2025-01-'+day)/'validation.npz')
        np.testing.assert_array_equal(q['event_indices'], ix)
        np.testing.assert_array_equal(q['labels'], d['labels'])
        add(name+'_delta', logit(q['predictions'].astype(np.float64).clip(1e-7, 1-1e-7))-z)
    x = np.stack(values, 2).reshape(-1, len(names)).astype('f')
    assert np.isfinite(x).all()
    return d, x, z.ravel(), names


def run():
    from ctr_data import load_cache
    from run import timestamp
    from arena_validation import importance_weights, weighted_auc
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    parent = (ROOT/pointer['csv']).parent
    # The probe's reference is frozen once, even if a later campaign advances.
    write_json(OUT/'incumbent_pointer.json', pointer)
    frozen = {pointer[k]:sha(ROOT/pointer[k]) for k in ['csv', 'manifest', 'acceptance']}
    sources = []
    for folder in ['final_v6', 'final_v8', 'final_v9']:
        m = json.loads((ROOT/'competition'/folder/'manifest.json').read_text())
        sources += [r['name'] for r in m['models']]
    sources = list(dict.fromkeys(sources))
    _, raw = load_cache(ROOT/'data/cache_v1', 'train')
    folds = {day:inputs(day, parent, sources, raw) for day in ['24', '25', '26']}
    d, x, offset, names = folds['24']
    fitting = np.flatnonzero(raw['timestamps'][d['event_indices']] < timestamp('2025-01-25'))
    assert np.unique(raw['users'][d['event_indices'][fitting]]).size == len(fitting)
    split = np.random.default_rng(8047).permutation(fitting)
    train_events, hold_events = split[len(split)//5:], split[:len(split)//5]

    def rows(events):
        return (events[:, None]*10+np.arange(10)).ravel()

    train_rows, hold_rows, all_rows = rows(train_events), rows(hold_events), rows(fitting)
    y = d['labels'].ravel()
    report = dict(reference=pointer, fitting_date='2025-01-24', fitting_events=len(fitting),
                  calibration_holdout_events=len(hold_events), sources=sources, feature_names=names, candidates=[],
                  promotion_ready=False, reason='Jan24 labels fit the calibrator. A proper earlier calibration fold is required for a fresh Jan24 audit.')
    for leaves in [7, 15]:
        params = dict(objective='binary', metric='auc', learning_rate=.02, num_leaves=leaves,
                      max_depth=4, min_data_in_leaf=1000, lambda_l2=100, feature_fraction=.8,
                      bagging_fraction=.8, bagging_freq=1, max_bin=63, num_threads=1,
                      force_col_wise=True, seed=8047, verbosity=-1, boost_from_average=False)
        started = time.monotonic()
        train = lgb.Dataset(x[train_rows], label=y[train_rows], init_score=offset[train_rows], params=params)
        valid = lgb.Dataset(x[hold_rows], label=y[hold_rows], init_score=offset[hold_rows], reference=train)
        model = lgb.train(params, train, num_boost_round=500, valid_sets=[valid],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        best = model.best_iteration or model.current_iteration()
        row = dict(leaves=leaves, fixed_iterations=best,
                   internal_base_auc=float(roc_auc_score(y[hold_rows], offset[hold_rows])),
                   internal_auc=float(roc_auc_score(y[hold_rows], offset[hold_rows]+model.predict(x[hold_rows], raw_score=True))))
        model = lgb.train(params, lgb.Dataset(x[all_rows], label=y[all_rows], init_score=offset[all_rows], params=params),
                          num_boost_round=best)
        model.save_model(str(OUT/('meta'+str(leaves)+'.txt')))
        for day in ['25', '26']:
            v, features, base, cols = folds[day]
            assert cols == names
            assert not np.intersect1d(d['event_indices'][fitting], v['event_indices']).size
            p = expit(base+model.predict(features, raw_score=True)).reshape(-1, 10)
            iw = importance_weights('2025-01-'+day, v['event_indices'])[0]
            old_auc = float(roc_auc_score(v['labels'].ravel(), v['predictions'].ravel()))
            auc = float(roc_auc_score(v['labels'].ravel(), p.ravel()))
            row[day] = dict(old_auc=old_auc, auc=auc, delta=auc-old_auc,
                           weighted_auc=weighted_auc(v['labels'], p, iw),
                           old_weighted_auc=weighted_auc(v['labels'], v['predictions'], iw))
            np.savez_compressed(OUT/('meta'+str(leaves)+'_validation_'+day+'.npz'), event_indices=v['event_indices'], labels=v['labels'], predictions=p)
        row['seconds'] = time.monotonic()-started
        report['candidates'].append(row)
        write_json(OUT/'report.json', report)
        print(json.dumps(row), flush=True)
    assert all(sha(ROOT/file) == value for file,value in frozen.items())


if __name__ == '__main__':
    run()
