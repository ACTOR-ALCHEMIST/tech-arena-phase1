"""Frozen-model temporal-feature diagnostics; no fitting or CSV changes."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from arena_ood import ROOT, arrays, write_json


def run():
    from arena_tabular import predict
    from arena_push_model import build_model
    from ctr_data import load_cache
    out = ROOT/'competition/time_ablation'
    out.mkdir(parents=True, exist_ok=True)
    _, raw = load_cache(ROOT/'data/cache_v1', 'train')
    torch.set_num_threads(1)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    report = dict(model='spectral_multi_wide', interpretation='Feature substitution importance; not an independently trained candidate or an estimate of achievable gain.', folds={})
    for day in ['26', '25']:
        cutoff = '2025-01-'+day
        _, info, labels, _, values = arrays(cutoff, 'spectral_multi')
        path = ROOT/'competition/ood/runs/spectral_multi_wide'/cutoff
        s = torch.load(path/'model.pt', map_location='cpu', weights_only=True)
        model = build_model(s['width'], s['state_dict']['mean'].numpy(), s['state_dict']['scale'].numpy(), s['config'])
        model.load_state_dict(s['state_dict'])
        model.to(device)
        names, ix = info['feature_names'], labels['validation_indices']
        x = values[:]
        saved = np.load(path/'validation.npz')['predictions']
        y = labels['valid'].ravel()
        hist, age, ts = raw['history'][ix], raw['age_hours'][ix], raw['timestamps'][ix]
        minutes = ts[:, None]//60-np.rint(age*60).astype(np.int64)
        bucket_known = (((minutes%1440)//360 == (ts%86400//21600)[:, None]) & (hist != 0)).any(1)
        weekend_known = (((((minutes//1440+3)%7) >= 5) == (raw['context'][ix, 2] > 0)[:, None]) & (hist != 0)).any(1)
        rows = []
        old_auc = float(roc_auc_score(y, saved.ravel()))
        for kind in ['identity', 'bucket_semantic', 'weekend_semantic', 'context_affinity', 'all_temporal']:
            v = x.copy()
            if kind in ['bucket_semantic', 'weekend_semantic', 'all_temporal']:
                suffixes = ['bucket', 'weekend'] if kind == 'all_temporal' else ['bucket' if kind == 'bucket_semantic' else 'weekend']
                for label in suffixes:
                    occupied = np.repeat(bucket_known if label == 'bucket' else weekend_known, 10)
                    for name in names:
                        suffix = '_hist_same_'+label+'_cos'
                        if name.endswith(suffix):
                            mean = name[:-len(suffix)]+'_hist_cos_mean'
                            col = names.index(name)
                            v[occupied, col] = v[occupied, names.index(mean)]
            if kind in ['context_affinity', 'all_temporal']:
                for name in ['user_bucket_affinity', 'user_weekend_affinity', 'user_bucket_ctr_offset',
                             'item_bucket_affinity', 'item_weekend_affinity', 'item_bucket_ctr_offset']:
                    v[:, names.index(name)] = 0
            p = predict(model, v, device, batch_events=1024)
            row = dict(condition=kind, auc=float(roc_auc_score(y, p.ravel())))
            row['delta'] = row['auc']-old_auc
            if kind == 'identity':
                row['replay_max_error'] = float(np.max(np.abs(saved-p)))
                assert row['replay_max_error'] < 2e-5
            rows.append(row)
            print(json.dumps(dict(cutoff=cutoff, **row)), flush=True)
        report['folds'][day] = rows
        write_json(out/'report.json', report)
        del x, model
        if device.type == 'mps':
            torch.mps.empty_cache()


if __name__ == '__main__':
    run()
