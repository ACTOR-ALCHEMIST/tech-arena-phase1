"""Small fixed-grid feedback corrections; freeze before a third-origin check."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from arena_ood import ROOT, write_json, sha

OUT = ROOT/'competition/pair_adjustment'


def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))


def corrections(data, indices, cutoff):
    from arena_pair_memory import memory, extra_features
    from arena_core import snapshot
    x, _ = extra_features(data, indices, cutoff)
    x = x.reshape(len(indices), 10, -1)
    m = memory(cutoff)
    stats = dict(np.load(snapshot(cutoff)/'stats.npz'))
    uid, iid = data['users'][indices], data['items'][indices]
    repeat = (iid[:, :, None] == data['history'][indices, None, :]).any(2)
    personal = logit(m['user_repeat_ctr'][uid])-logit(m['user_nonrepeat_ctr'][uid])
    personal -= logit(m['user_repeat_ctr'][0])-logit(m['user_nonrepeat_ctr'][0])
    item = logit(m['item_repeat_ctr'][iid])-logit(m['item_repeat_ctr'][0])-stats['ib'][iid]
    preference = (personal.clip(-2, 2)[:, None]+item.clip(-2, 2))*repeat
    return np.stack([x[:, :, 5], x[:, :, 8], preference], -1).astype('f')


def apply(p, features, weights):
    if not any(weights):
        return p.copy()
    z = logit(p.astype(np.float64))+(features*np.array(weights)).sum(2)
    return 1/(1+np.exp(-np.clip(z, -40, 40)))


def run():
    from ctr_data import load_cache
    from arena_validation import importance_weights, weighted_auc
    from arena_push084_audit import compare
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    if (OUT/'incumbent_pointer.json').exists():
        pointer = json.loads((OUT/'incumbent_pointer.json').read_text())
    else:
        write_json(OUT/'incumbent_pointer.json', pointer)
    parent = (ROOT/pointer['csv']).parent
    _, raw = load_cache(ROOT/'data/cache_v1', 'train')
    groups = {}
    for day in ['26', '25']:
        d = np.load(parent/('validation_'+day+'.npz'))
        f = corrections(raw, d['event_indices'], '2025-01-'+day)
        groups[day] = (d, f, importance_weights('2025-01-'+day, d['event_indices'])[0])
    weights = [[0., 0., 0.]]
    for j, masses in enumerate([[.05, .1, .25, .5], [.25, .5, 1.], [.05, .1, .25, .5]]):
        for mass in masses:
            w = [0., 0., 0.];w[j] = mass;weights.append(w)
    weights += [[a,b,c] for a in [.1,.25] for b in [.25,.5] for c in [.1,.25]]
    rows = []
    for w in weights:
        row = dict(weights=w)
        for day, (d,f,iw) in groups.items():
            p = apply(d['predictions'], f, w)
            row['auc_'+day] = float(roc_auc_score(d['labels'].ravel(), p.ravel()))
            row['weighted_auc_'+day] = weighted_auc(d['labels'],p,iw)
        row['selection_score'] = .5*(row['weighted_auc_26']+row['weighted_auc_25'])
        rows.append(row)
    old = rows[0]
    eligible = [r for r in rows[1:] if all(r[k]>old[k]+1e-6 for k in ['auc_26','auc_25','selection_score'])]
    winner = max(eligible,key=lambda r:r['selection_score']) if eligible else old
    write_json(OUT/'selection.json',dict(incumbent=old,selected=winner,recipes=rows,eligible=len(eligible),
               features=['observed_pair_ctr_logit_delta','prior_click_outside_visible_history','repeat_preference_logit_delta'],
               promoted=False,baseline=pointer,third_used_for_selection=False))
    for day, (d,f,_) in groups.items():
        p = apply(d['predictions'],f,winner['weights'])
        np.savez_compressed(OUT/('validation_'+day+'.npz'),event_indices=d['event_indices'],labels=d['labels'],predictions=p)
    print(json.dumps(dict(stage='feedback_correction_selection',old=old,new=winner,eligible=len(eligible))),flush=True)
    if eligible:
        old24 = np.load(parent/'validation_24.npz')
        f = corrections(raw,old24['event_indices'],'2025-01-24')
        new24 = dict(event_indices=old24['event_indices'],labels=old24['labels'],predictions=apply(old24['predictions'],f,winner['weights']))
        np.savez_compressed(OUT/'validation_24.npz',**new24)
        audit = compare(old24,new24,'2025-01-24')
        audit['selection_sha256'] = sha(OUT/'selection.json')
        write_json(OUT/'audit_24.json',audit)
        print(json.dumps(dict(stage='feedback_correction_third_origin',audit=audit)),flush=True)


if __name__ == '__main__':
    run()
