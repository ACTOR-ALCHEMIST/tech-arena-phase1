"""Read-only test of regularized conditional user likelihood in fixed SGNS8.

Uses only the fitting snapshot's binary positive matrix. A fixed quadrature
sample approximates the item partition function; validation labels are used
only for reporting, never for fitting the user posteriors.
"""
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from arena_ood import ROOT, write_json


def fit(v, bias, x, initial, ridge, sample=2048, steps=10):
    n = np.diff(x.indptr).astype(np.float64)
    sums = np.asarray(x@v, dtype=np.float64)
    probability = np.exp(bias.astype(np.float64)-float(bias.max()))
    probability[0] = 0
    probability /= probability.sum()
    ids = np.random.default_rng(1804).choice(len(v), sample, p=probability)
    q = v[ids].astype(np.float64)
    outer = (q[:, :, None]*q[:, None, :]).reshape(len(q), -1)
    u = initial.copy().astype(np.float64)
    eye = np.eye(v.shape[1])
    for _ in range(steps):
        for start in range(0, len(u), 512):
            end = min(start+512, len(u))
            a, counts = u[start:end], n[start:end]
            p = softmax(a@q.T, axis=1)
            mean = p@q
            cov = (p@outer).reshape(-1, v.shape[1], v.shape[1])-mean[:, :, None]*mean[:, None, :]
            hessian = cov*counts[:, None, None]+ridge*eye
            gradient = sums[start:end]-counts[:, None]*mean-ridge*a
            delta = np.linalg.solve(hessian, gradient[..., None])[..., 0]
            delta *= np.minimum(1, .5/np.linalg.norm(delta, axis=1).clip(1e-12))[:, None]
            u[start:end] += delta
        u[n == 0] = 0
    return u.astype('f')


def run():
    from arena_core import snapshot
    from arena_sgns_features import embedding_model
    from ctr_data import load_cache, temporal_split
    from run import timestamp
    from sklearn.metrics import roc_auc_score
    from arena_finalize import blend
    out = ROOT/'competition/posterior_probe'
    out.mkdir(parents=True, exist_ok=True)
    _, raw = load_cache(ROOT/'data/cache_v1', 'train')
    pointer = json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    parent = (ROOT/pointer['csv']).parent
    reports = {}
    with threadpool_limits(limits=2, user_api='blas'):
        for cutoff in ['2025-01-26', '2025-01-25']:
            model = embedding_model(cutoff, 8, recipe=dict(epochs=320, lr=.01, lr_decay=False, seed=2026,
                                      reuse_experiment='sgns8_const_e320_s2026'))
            x = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
            _, indices = temporal_split(raw, timestamp(cutoff))
            old = np.load(parent/('validation_'+cutoff[-2:]+'.npz'))
            np.testing.assert_array_equal(indices, old['event_indices'])
            users, items = raw['users'][indices], raw['items'][indices]
            labels = raw['labels'][indices]
            base_dot = np.einsum('bik,bk->bi', model['items'][items], model['users'][users])
            rows = [dict(kind='original_dot', auc=float(roc_auc_score(labels.ravel(), base_dot.ravel())))]
            for ridge in [1., 4., 16.]:
                path = out/(cutoff+'_ridge'+str(ridge)+'.npz')
                started = time.monotonic()
                if path.exists():
                    u = np.load(path)['users']
                else:
                    u = fit(model['items'], model['bias'], x, model['users'], ridge)
                    np.savez_compressed(path, users=u)
                dot = np.einsum('bik,bk->bi', model['items'][items], u[users])
                row = dict(kind='posterior', ridge=ridge, dot_auc=float(roc_auc_score(labels.ravel(), dot.ravel())),
                           seconds=time.monotonic()-started)
                # A few predeclared, small residual corrections; no promotion.
                std = float(base_dot.std())
                base_logit = np.log(old['predictions'].clip(1e-7, 1-1e-7)/(1-old['predictions'].clip(1e-7, 1-1e-7)))
                row['corrections'] = {str(w):float(roc_auc_score(labels.ravel(), (base_logit+w*(dot-base_dot)/std).ravel())) for w in [.05, .1, .25]}
                rows.append(row)
                print(json.dumps(dict(cutoff=cutoff, **row)), flush=True)
            reports[cutoff] = rows
            write_json(out/'report.json', dict(incumbent=pointer, origins=reports, note='Approximate partition, fixed quadrature; no current-event labels used in posterior fit; diagnostic only.'))


if __name__ == '__main__':
    run()
