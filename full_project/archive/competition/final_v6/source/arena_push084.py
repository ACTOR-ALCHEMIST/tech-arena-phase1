"""New representations campaign; immutable incumbent and locked second cutoff."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from arena_ood import ROOT, guard, sha, write_json

OUT = ROOT/'competition/push084'
NAMES = ['coclick_cross', 'coclick_wide', 'exposure_cross', 'exposure_wide']


def command(args, label):
    write_json(OUT/'state.json', dict(stage=label, pid=os.getpid(), updated=time.time(), command=args))
    print(json.dumps(dict(stage=label, command=args)), flush=True)
    with (OUT/(label+'.log')).open('a') as log:
        subprocess.run([sys.executable, '-u']+args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def record(name, cutoff):
    path = ROOT/'competition/ood/runs'/name/cutoff
    return json.loads((path/'result.json').read_text()), np.load(path/'validation.npz')


def select():
    from arena_finalize import blend
    from arena_validation import importance_weights, weighted_auc
    from sklearn.metrics import roc_auc_score
    pointer = json.loads((OUT/'incumbent_pointer.json').read_text())
    parent = (ROOT/pointer['csv']).parent
    groups = {}
    for cutoff in ['2025-01-26', '2025-01-25']:
        base = np.load(parent/('validation_'+cutoff[-2:]+'.npz'))
        parts = [base['predictions']]
        for name in NAMES:
            _, d = record(name, cutoff)
            for k in ['event_indices', 'labels']:
                np.testing.assert_array_equal(d[k], base[k])
            parts.append(d['predictions'])
        refreshed = np.load(ROOT/'competition/robust_mf/diagnostics'/('refreshed_'+cutoff[-2:]+'.npz'))
        for k in ['event_indices', 'labels']:
            np.testing.assert_array_equal(refreshed[k], base[k])
        parts.append(refreshed['predictions'])
        groups[cutoff] = (base, parts, importance_weights(cutoff, base['event_indices'])[0])
    sources = ['incumbent']+NAMES+['refreshed_mf']
    weights = [[1.]+[0.]*(len(sources)-1)]
    for j in range(1, len(sources)):
        for mass in [.05, .1, .25, .5, .75, 1.]:
            w = [1-mass]+[0.]*(len(sources)-1)
            w[j] = mass
            weights.append(w)
    families = [[1, 2], [3, 4], [1, 2, 3, 4], list(range(1, len(NAMES)+1))]
    if len(NAMES) >= 6:
        families += [[5, 6], [1, 2, 5, 6]]
    if len(NAMES) >= 10:
        families += [[1, 2, 3, 4, 7, 8, 9, 10]]
    if len(NAMES) >= 12:
        families += [[11, 12], [1, 2, 3, 4, 11, 12]]
    if len(NAMES) >= 14:
        families += [[13, 14], [1, 2, 3, 4, 7, 11, 12, 13, 14]]
    graph_indices = [i+1 for i, name in enumerate(NAMES) if 'spectral' in name]
    if len(graph_indices) > 1:
        families += [graph_indices, [1, 2, 3, 4, 6]+graph_indices]
    for indices in families:
        for mass in [.25, .5, .75, 1.]:
            w = [1-mass]+[0.]*(len(sources)-1)
            for j in indices:
                w[j] = mass/len(indices)
            weights.append(w)
    rows = []
    for w in weights:
        row = dict(weights=w, kind='logit')
        for cutoff, (d, parts, iw) in groups.items():
            p = blend(parts, w, 'logit')
            row['auc_'+cutoff[-2:]] = float(roc_auc_score(d['labels'].ravel(), p.ravel()))
            row['weighted_auc_'+cutoff[-2:]] = weighted_auc(d['labels'], p, iw)
        row['selection_score'] = .5*(row['weighted_auc_26']+row['weighted_auc_25'])
        rows.append(row)
    old = rows[0]
    eligible = [r for r in rows[1:] if all(r[k] > old[k]+1e-6 for k in ['auc_26', 'auc_25', 'selection_score'])]
    selected = max(eligible, key=lambda r: r['selection_score']) if eligible else old
    result = dict(sources=sources, incumbent=old, selected=selected, eligible=len(eligible), recipes=rows,
                  candidates=[dict(name=n, primary=record(n, '2025-01-26')[0], secondary=record(n, '2025-01-25')[0]) for n in NAMES])
    write_json(OUT/'selection.json', result)
    for cutoff, (d, parts, _) in groups.items():
        np.savez_compressed(OUT/('selected_validation_'+cutoff[-2:]+'.npz'), event_indices=d['event_indices'],
                            labels=d['labels'], predictions=blend(parts, selected['weights'], 'logit'))
    print(json.dumps(dict(stage='selection', old=old, selected=selected, eligible=len(eligible))), flush=True)
    return result


def train():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if '--wait' in sys.argv else fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        before = guard()
        if (OUT/'incumbent.json').exists():
            assert json.loads((OUT/'incumbent.json').read_text()) == before
        else:
            write_json(OUT/'incumbent.json', before)
            shutil.copy2(ROOT/'competition/CURRENT_BEST.json', OUT/'incumbent_pointer.json')
        for name in NAMES:
            args = ['arena_ood.py', 'train', '--name', name, '--config', 'configs/'+name+'.json']
            primary_path = ROOT/'competition/ood/runs'/name/'2025-01-26/result.json'
            if not primary_path.exists():
                command(args, name+'_26')
            else:
                assert json.loads(primary_path.read_text())['augmentation_config'] == json.loads((ROOT/('configs/'+name+'.json')).read_text())
            r, _ = record(name, '2025-01-26')
            secondary_path = ROOT/'competition/ood/runs'/name/'2025-01-25/result.json'
            if not secondary_path.exists():
                command(args+['--cutoff', '2025-01-25', '--fixed-epochs', str(r['best_epoch'])], name+'_25')
            else:
                assert json.loads(secondary_path.read_text())['best_epoch'] == r['best_epoch']
        assert guard() == before
        select()
        write_json(OUT/'state.json', dict(stage='training_completed', pid=os.getpid(), updated=time.time()))


if __name__ == '__main__':
    if '--daily' in sys.argv:
        NAMES += ['daily_coclick_cross', 'daily_coclick_wide']
    if '--extra' in sys.argv:
        NAMES += ['rich_coclick_wide', 'balanced_coclick_wide', 'rank_coclick_wide', 'spectral_wide']
    if '--softmax' in sys.argv:
        NAMES += ['softmax30_wide', 'softmax90_wide']
    if '--gate' in sys.argv:
        NAMES += ['gate_coclick_wide', 'gate_rich_cross']
    if '--joint' in sys.argv:
        NAMES += ['joint_graph_wide']
    if '--weekend' in sys.argv:
        NAMES += ['weekend_wide']
    if '--graph64' in sys.argv:
        NAMES += ['spectral64_wide', 'spectral_multi_wide']
    if '--normal' in sys.argv:
        NAMES += ['normalized_spectral_wide']
    if '--origins' in sys.argv:
        NAMES += ['origin_coclick_wide', 'origin_spectral_wide']
    if '--bias' in sys.argv:
        NAMES += ['bias80_wide', 'bias240_wide']
    train()
