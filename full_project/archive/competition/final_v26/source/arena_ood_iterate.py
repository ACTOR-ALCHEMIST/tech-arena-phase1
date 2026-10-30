"""Execute a declared augmentation pool and locked rolling-origin audits."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import numpy as np
from arena_ood import ROOT, OUT, RECIPES, guard, write_json


def record(name, cutoff):
    path = OUT/'runs'/name/cutoff
    return json.loads((path/'result.json').read_text()), np.load(path/'validation.npz')


def select(names):
    from arena_finalize import blend
    from arena_validation import importance_weights, weighted_auc
    from sklearn.metrics import roc_auc_score
    groups = {}
    for cutoff in ['2025-01-26', '2025-01-25']:
        baseline = np.load(ROOT/'competition/v3'/('selected_validation_'+cutoff[-2:]+'.npz'))
        predictions = [baseline['predictions']]
        for name in names:
            _, d = record(name, cutoff)
            for k in ['labels', 'event_indices']:
                np.testing.assert_array_equal(baseline[k], d[k])
            predictions.append(d['predictions'])
        iw, profile = importance_weights(cutoff, baseline['event_indices'])
        groups[cutoff] = (baseline['labels'], baseline['event_indices'], predictions, iw, profile)
    weights = [[1.]+[0.]*len(names)]
    for i in range(len(names)):
        for mass in [.05, .1, .25, .5, 1.]:
            w = [1-mass]+[0.]*len(names)
            w[i+1] = mass
            weights.append(w)
    for mass in [.25, .5]:
        weights.append([1-mass]+[mass/len(names)]*len(names))
    rows = []
    for w in weights:
        row = {'weights': w, 'kind': 'logit'}
        for cutoff, (y, indices, pred, iw, _) in groups.items():
            p = blend(pred, w, 'logit')
            row['auc_'+cutoff[-2:]] = float(roc_auc_score(y.ravel(), p.ravel()))
            row['weighted_auc_'+cutoff[-2:]] = weighted_auc(y, p, iw)
        row['selection_score'] = .5*(row['weighted_auc_26']+row['weighted_auc_25'])
        rows.append(row)
    old = rows[0]
    eligible = [r for r in rows[1:] if all(r[k] > old[k]+1e-6 for k in ['auc_26', 'auc_25', 'selection_score'])]
    winner = max(eligible, key=lambda r: r['selection_score']) if eligible else old
    result = dict(sources=['frozen_v3']+names, selected=winner, incumbent=old, eligible=len(eligible), recipes=rows,
                  candidates=[dict(name=name, primary=record(name, '2025-01-26')[0], secondary=record(name, '2025-01-25')[0]) for name in names],
                  protocol='Clean Jan26 model selection; Jan25 epoch/schedule locked. Coarse logit mixtures must improve clean AUC at both cutoffs and mean cohort-weighted AUC. No evaluation labels.')
    write_json(OUT/'selection.json', result)
    for cutoff, (y, indices, pred, _, _) in groups.items():
        np.savez_compressed(OUT/('selected_validation_'+cutoff[-2:]+'.npz'), labels=y, event_indices=indices,
                            predictions=blend(pred, winner['weights'], 'logit'))
    print(json.dumps({'stage': 'selection', 'selected': winner, 'incumbent': old, 'eligible': len(eligible)}), flush=True)
    return result


def run(names):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.iteration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        before = guard()
        if (OUT/'incumbent.json').exists():
            assert json.loads((OUT/'incumbent.json').read_text()) == before
        else:
            write_json(OUT/'incumbent.json', before)
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        for name in names:
            for cutoff in ['2025-01-26', '2025-01-25']:
                args = [sys.executable, '-u', 'arena_ood.py', 'train', '--name', name, '--cutoff', cutoff]
                if cutoff.endswith('25'):
                    args += ['--fixed-epochs', str(record(name, '2025-01-26')[0]['best_epoch'])]
                write_json(OUT/'state.json', dict(stage='training', name=name, cutoff=cutoff, pid=os.getpid(), started=time.time()))
                with (OUT/(name+'_'+cutoff[-2:]+'.log')).open('a') as log:
                    subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                assert guard() == before
        select(names)
        write_json(OUT/'state.json', dict(stage='experiments_completed', pid=os.getpid(), finished=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--names', nargs='+', default=list(RECIPES))
    p.add_argument('--select-only', action='store_true')
    a = p.parse_args()
    select(a.names) if a.select_only else run(a.names)
