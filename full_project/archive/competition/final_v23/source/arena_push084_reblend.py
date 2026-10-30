"""A small, predefined two-step refinement before the untouched third origin."""
import copy
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from arena_ood import ROOT, write_json
from arena_push084 import OUT


def run():
    from arena_finalize import blend
    from arena_validation import importance_weights, weighted_auc
    selection = json.loads((OUT/'selection.json').read_text())
    if selection.get('refinement'):
        return
    write_json(OUT/'static_selection.json', selection)
    original = copy.deepcopy(selection['selected'])
    sources, groups = selection['sources'], {}
    pointer = json.loads((OUT/'incumbent_pointer.json').read_text())
    folder = (ROOT/pointer['csv']).parent
    for day in ['26', '25']:
        data = np.load(folder/('validation_'+day+'.npz'))
        parts = [data['predictions']]
        for name in sources[1:]:
            path = ROOT/'competition/robust_mf/diagnostics'/('refreshed_'+day+'.npz') if name == 'refreshed_mf' else ROOT/'competition/ood/runs'/name/('2025-01-'+day)/'validation.npz'
            d = np.load(path)
            for key in ['event_indices', 'labels']:
                np.testing.assert_array_equal(d[key], data[key])
            parts.append(d['predictions'])
        groups[day] = (data, parts, importance_weights('2025-01-'+day, data['event_indices'])[0])
    base = selection['incumbent']
    ranked = {}
    for r in selection['recipes']:
        active = [j for j, w in enumerate(r['weights']) if j > 0 and w > 0]
        if len(active) == 1 and all(r[k] > base[k]+1e-6 for k in ['auc_26', 'auc_25', 'selection_score']):
            j = active[0]
            ranked[j] = max(ranked.get(j, -np.inf), r['selection_score'])
    pool = []
    for j in sorted(ranked, key=ranked.get, reverse=True)[:5]:
        w = np.zeros(len(sources));w[j] = 1
        pool.append((sources[j], w))
    first_four = np.zeros(len(sources));first_four[1:5] = .25
    pool.append(('mean_first_four', first_four))
    chosen = selection['selected']
    for step in range(2):
        best = chosen
        for name, candidate in pool:
            for mass in [.1, .25, .5]:
                w = (1-mass)*np.array(chosen['weights'])+mass*candidate
                row = dict(weights=w.tolist(), kind='logit', refinement_step=step+1, added=name, added_weight=mass)
                for day, (d, parts, iw) in groups.items():
                    p = blend(parts, w, 'logit')
                    row['auc_'+day] = float(roc_auc_score(d['labels'].ravel(), p.ravel()))
                    row['weighted_auc_'+day] = weighted_auc(d['labels'], p, iw)
                row['selection_score'] = .5*(row['weighted_auc_26']+row['weighted_auc_25'])
                selection['recipes'].append(row)
                if row['selection_score'] > best['selection_score']+1e-5 and all(row[k] > chosen[k]+1e-6 for k in ['auc_26', 'auc_25']):
                    best = row
        if best is chosen:
            break
        chosen = best
        print(json.dumps(dict(stage='weight_refinement', step=step+1, selected=chosen)), flush=True)
    selection['selected'] = chosen
    selection['refinement'] = dict(static_selected=original, max_steps=2, pool=[n for n, _ in pool],
                                   masses=[.1, .25, .5], minimum_score_gain=1e-5, third_origin_used=False)
    write_json(OUT/'selection.json', selection)
    for day, (d, parts, _) in groups.items():
        np.savez_compressed(OUT/('selected_validation_'+day+'.npz'), event_indices=d['event_indices'], labels=d['labels'], predictions=blend(parts, chosen['weights'], 'logit'))
    print(json.dumps(dict(stage='final_weight_selection', selected=chosen)), flush=True)


if __name__ == '__main__':
    run()
