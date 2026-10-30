"""Full-window refit, raw-input replay, and separately accepted V4 submission."""
import argparse
import copy
import csv
import json
import os
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from arena_ood import ROOT, OUT, sha, guard, write_json, source, corrupt

FINAL = ROOT/'competition/final_v4'


def final_inputs(expanded=False, rolling_days=None, origin_days=None):
    from ctr_data import load_cache
    from arena_finalize import feature_cache, write_feature_matrix
    from arena_ood_features import extension_features, FeatureMatrix
    from run import timestamp
    base = source('2025-01-26')
    fc = base['feature_config']
    _, data = load_cache(ROOT/'data/cache_v1', 'train')
    _, evaluation = load_cache(ROOT/'data/cache_v1', 'evaluation')
    frame_training = bool(rolling_days or origin_days)
    if origin_days:
        from arena_origins import training_inputs
        train, indices, rolling_names = training_inputs('2025-01-27', expanded, origin_days)
    elif rolling_days:
        from arena_daily import training_inputs
        train, indices, rolling_names = training_inputs('2025-01-27', expanded, rolling_days)
    else:
        indices = np.flatnonzero(data['timestamps'] >= timestamp(fc['train_start']))
        train = write_feature_matrix(data, indices, fc['train_start'], fc, feature_cache(fc, 'train'))
    valid = write_feature_matrix(evaluation, np.arange(len(evaluation['users'])), '2025-01-27', fc, feature_cache(fc, 'evaluation'))
    names = json.loads((ROOT/base['feature_dataset']/'metadata.json').read_text())['feature_names']
    extensions = [True, 'coclick'] if expanded == 'coclick_rich' else ['spectral', 'spectral64'] if expanded == 'spectral_multi' else [expanded[5:], 'pair_memory'] if expanded in ['pair_coclick', 'pair_spectral'] else [expanded] if expanded else []
    for extension in extensions:
        folder = OUT/('final_inputs_'+extension if isinstance(extension, str) else 'final_inputs')
        folder.mkdir(parents=True, exist_ok=True)
        for phase, raw, ix, frame in [('train', data, indices, fc['train_start']),
                                      ('evaluation', evaluation, np.arange(len(evaluation['users'])), '2025-01-27')]:
            if phase == 'train' and frame_training:
                continue
            if not (folder/(phase+'.json')).exists():
                x, cols = extension_features(raw, ix, frame, fc, extension)
                np.save(folder/(phase+'.npy'), x)
                write_json(folder/(phase+'.json'), {'names': cols, 'events': len(ix), 'frame': frame})
        cols = json.loads((folder/'evaluation.json').read_text())['names']
        if not frame_training:
            assert cols == json.loads((folder/'train.json').read_text())['names']
        names += cols
        if not frame_training:
            train = FeatureMatrix(train, np.load(folder/'train.npy', mmap_mode='r'))
        valid = FeatureMatrix(valid, np.load(folder/'evaluation.npy', mmap_mode='r'))
    if origin_days:
        from arena_origins import delay, DELAY_NAME
        valid = FeatureMatrix(valid, delay(evaluation, np.arange(len(evaluation['users'])), '2025-01-27'))
        names += [DELAY_NAME]
    if frame_training:
        assert rolling_names == names
    return base, data, evaluation, indices, train, valid, names


def fit(name):
    from arena_tabular import predict
    from arena_push_model import build_model, normalizer
    from arena_features import complete_features
    from arena_ood_features import extension_features
    r = json.loads((OUT/'runs'/name/'2025-01-26'/'result.json').read_text())
    aug = r['augmentation_config']
    cfg = r['config']
    path = FINAL/'models'/name
    path.mkdir(parents=True, exist_ok=True)
    if (path/'recipe.json').exists():
        return path
    extension = aug.get('feature_mode', aug.get('expanded', False))
    from arena_windows import days
    rolling_days, origin_days = days(aug, '2025-01-27', 'rolling'), days(aug, '2025-01-27', 'origin')
    base, data, evaluation, indices, x, v, names = final_inputs(extension, rolling_days, origin_days)
    augmented = None
    if aug.get('mix', 0):
        # The final augmentation view is ephemeral; inference needs clean inputs.
        selected = np.sort(np.random.default_rng(61973).choice(len(indices), round(len(indices)*.5), replace=False))
        lookup = np.full(len(indices), -1, np.int64)
        lookup[selected] = np.arange(len(selected))
        inputs = {k: np.array(value[indices[selected]]) for k, value in data.items() if k != 'labels'}
        inputs = corrupt(inputs, aug['augmentation'], 61974)
        augmented, columns = complete_features(inputs, np.arange(len(selected)), base['feature_config']['train_start'], base['feature_config'])
        if extension:
            extra, cols = extension_features(inputs, np.arange(len(selected)), base['feature_config']['train_start'], base['feature_config'], extension)
            augmented = np.concatenate([augmented, extra], axis=1)
            columns += cols
        assert columns == names
        conditional = aug['mix']/(len(selected)/len(indices))
        del inputs
    mean, scale = normalizer(x, cfg)
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = build_model(x.shape[1], mean, scale, cfg).to(device)
    ema = copy.deepcopy(model).eval() if cfg.get('ema') else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['lr_schedule_epochs'], eta_min=.0001)
    rng = np.random.default_rng(cfg['seed'])
    aug_rng = np.random.default_rng(cfg['seed']+7331)
    labels = data['labels'][indices]
    from arena_objectives import loss as objective, user_weights
    event_weights = user_weights(data['users'][indices], cfg.get('user_balance', 0))
    records = []
    for epoch in range(1, r['best_epoch']+1):
        tick = time.monotonic()
        order = rng.permutation(len(indices))
        model.train()
        total, changed = 0., 0
        for offset in range(0, len(order), cfg.get('batch_events', 1024)):
            chosen = order[offset:offset+cfg.get('batch_events', 1024)]
            rows = (chosen[:, None]*10+np.arange(10)).ravel()
            values = np.array(x[rows]).reshape(-1, 10, x.shape[1])
            if augmented is not None:
                positions = lookup[chosen]
                selected = (positions >= 0) & (aug_rng.random(len(chosen)) < conditional)
                if selected.any():
                    aug_rows = (positions[selected, None]*10+np.arange(10)).ravel()
                    values[selected] = augmented[aug_rows].reshape(-1, 10, x.shape[1])
                    changed += int(selected.sum())
            bx = torch.tensor(values, device=device)
            by = torch.tensor(labels[chosen], dtype=torch.float32, device=device)
            z = model(bx)
            bw = torch.tensor(event_weights[chosen], device=device) if event_weights is not None else None
            loss = objective(z, by, cfg, bw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item()*len(chosen)
            if ema is not None:
                with torch.no_grad():
                    for target, parameter in zip(ema.parameters(), model.parameters()):
                        target.mul_(cfg['ema']).add_(parameter, alpha=1-cfg['ema'])
        scheduler.step()
        row = dict(epoch=epoch, train_loss=total/len(indices), augmented_events=changed, seconds=time.monotonic()-tick)
        records.append(row)
        print(json.dumps(dict(name=name, phase='final_fit', **row)), flush=True)
        write_json(path/'training.json', records)
    trained = ema if ema is not None else model
    torch.save({'state_dict': trained.state_dict(), 'width': x.shape[1], 'loss': cfg['loss'], 'config': cfg}, path/'model.pt')
    p = predict(trained, v, device)
    np.save(path/'evaluation_predictions.npy', p)
    write_json(path/'recipe.json', dict(name=name, model_config=cfg, augmentation_config=aug, feature_config=base['feature_config'],
               feature_names=names, training_events=len(indices), fixed_epochs=r['best_epoch'],
               unique_training_events=len(np.unique(indices)), rolling_days=rolling_days, origin_days=origin_days,
               feature_snapshot=base['feature_config']['train_start'], inference_snapshot='2025-01-27',
               training_indices_sha256=__import__('hashlib').sha256(indices.tobytes()).hexdigest()))
    return path


def csv_probabilities(path, expected_ids):
    values = []
    with path.open(newline='') as f:
        reader = csv.reader(f)
        assert next(reader) == ['rowid', 'predicted_probability']
        for i, (rid, p) in enumerate(reader):
            assert rid == expected_ids[i].decode()
            values.append(float(p))
    assert len(values) == len(expected_ids)
    return np.array(values).reshape(-1, 10)


def export():
    from arena_finalize import blend, export_csv
    from ctr_data import load_cache
    before = guard()
    assert before == json.loads((OUT/'incumbent.json').read_text())
    if (FINAL/'manifest.json').exists():
        return
    selection = json.loads((OUT/'selection.json').read_text())
    spec = selection['selected']
    FINAL.mkdir(parents=True, exist_ok=True)
    _, evaluation = load_cache(ROOT/'data/cache_v1', 'evaluation')
    shutil.copy2(ROOT/'competition/final_v3/submission.csv', FINAL/'frozen_v3.csv')
    predictions = [csv_probabilities(FINAL/'frozen_v3.csv', evaluation['rowids'].ravel())]
    weights = [spec['weights'][0]]
    models = []
    for name, weight in zip(selection['sources'][1:], spec['weights'][1:]):
        if weight > 0:
            path = fit(name)
            predictions.append(np.load(path/'evaluation_predictions.npy'))
            weights.append(weight)
            models.append(dict(name=name, weight=weight, path=str(path.relative_to(ROOT)),
                               model_sha256=sha(path/'model.pt'), recipe_sha256=sha(path/'recipe.json'),
                               predictions_sha256=sha(path/'evaluation_predictions.npy')))
    p = blend(predictions, weights, spec['kind']) if models else predictions[0]
    if models:
        export_csv(FINAL/'submission.csv', evaluation, p)
    else:
        shutil.copy2(FINAL/'frozen_v3.csv', FINAL/'submission.csv')
    np.save(FINAL/'predictions.npy', p)
    shutil.copy2(OUT/'selection.json', FINAL/'selection.json')
    for cutoff in ['26', '25']:
        shutil.copy2(OUT/('selected_validation_'+cutoff+'.npz'), FINAL/('validation_'+cutoff+'.npz'))
        if (OUT/('audit_'+cutoff+'.json')).exists():
            shutil.copy2(OUT/('audit_'+cutoff+'.json'), FINAL/('audit_'+cutoff+'.json'))
    if (OUT/'audit_24.json').exists():
        shutil.copy2(OUT/'audit_24.json', FINAL/'audit_24.json')
        shutil.copy2(OUT/'audited_validation_24.npz', FINAL/'validation_24.npz')
    if (OUT/'robustness_curve.png').exists():
        shutil.copy2(OUT/'robustness_curve.png', FINAL/'robustness_curve.png')
    # Freeze every inference representation so later experiments cannot alter replay.
    from arena_core import snapshot
    representations = FINAL/'representations'/'snapshot_2025-01-27'
    if not representations.exists():
        shutil.copytree(snapshot('2025-01-27'), representations)
    files = ['arena_ood.py', 'arena_ood_features.py', 'arena_ood_consensus.py', 'arena_ood_context.py', 'arena_ood_audit.py', 'arena_ood_iterate.py', 'arena_ood_export.py', 'arena_ood_finish.py', 'test_arena_ood.py',
             'arena_core.py', 'arena_features.py', 'arena_sgns_features.py', 'arena_sgns.py', 'arena_latent.py', 'arena_feedback.py',
             'arena_tabular.py', 'arena_neural_v2.py', 'arena_finalize.py', 'arena_models.py', 'arena_report.py', 'ctr_data.py', 'run.py', 'requirements-v3.txt']
    source_dir = FINAL/'source'
    source_dir.mkdir(exist_ok=True)
    for file in files:
        shutil.copy2(ROOT/file, source_dir/file)
    artifacts = {str(path.relative_to(FINAL)): sha(path) for path in FINAL.rglob('*')
                 if path.is_file() and path.name not in ['manifest.json', 'acceptance.json'] and not path.name.endswith('.lock')}
    raw_inputs = {str(path.relative_to(ROOT)): sha(path) for path in (ROOT/'data/cache_v1/evaluation').glob('*.npy')}
    manifest = dict(method='v4_history_augmentation_and_expanded_matching', selection=spec, sources=selection['sources'], models=models,
                    baseline_weight=spec['weights'][0], baseline_sha256=sha(FINAL/'frozen_v3.csv'), incumbent_digest=before,
                    csv='competition/final_v4/submission.csv', sha256=sha(FINAL/'submission.csv'), rows=len(p)*10,
                    validation_26={'auc': spec['auc_26']}, validation_25={'auc': spec['auc_25']},
                    artifact_sha256=artifacts, raw_input_sha256=raw_inputs, official_evaluation_labels_available=False,
                    target_auc=.835, target_reached_on_primary_validation=spec['auc_26'] >= .835)
    assert guard() == before
    write_json(FINAL/'manifest.json', manifest)


def verify():
    from ctr_data import load_cache
    from arena_features import complete_features
    from arena_ood_features import extension_features
    from arena_ood_audit import neural
    from arena_finalize import blend
    from run import check_submission
    manifest = json.loads((FINAL/'manifest.json').read_text())
    for file, digest in manifest['incumbent_digest'].items():
        if file != 'competition/CURRENT_BEST.json':
            assert sha(ROOT/file) == digest
    for file, digest in manifest['artifact_sha256'].items():
        assert sha(FINAL/file) == digest, file
    for file, digest in manifest['raw_input_sha256'].items():
        assert sha(ROOT/file) == digest, file
    check_submission(FINAL/'submission.csv', ROOT/'evaluation.zip')
    _, evaluation = load_cache(ROOT/'data/cache_v1', 'evaluation')
    expected = np.load(FINAL/'predictions.npy')
    written = csv_probabilities(FINAL/'submission.csv', evaluation['rowids'].ravel())
    rounding_error = float(np.max(np.abs(written-expected)))
    assert rounding_error < 1e-9
    parts = [csv_probabilities(FINAL/'frozen_v3.csv', evaluation['rowids'].ravel())]
    weights, errors = [manifest['baseline_weight']], []
    os.environ['ARENA_SNAPSHOT_ROOT'] = str(FINAL/'representations')
    torch.set_num_threads(2)
    for row in manifest['models']:
        path = ROOT/row['path']
        recipe = json.loads((path/'recipe.json').read_text())
        ix = np.arange(len(evaluation['users']))
        x, names = complete_features(evaluation, ix, '2025-01-27', recipe['feature_config'])
        extension = recipe['augmentation_config'].get('feature_mode', recipe['augmentation_config'].get('expanded', False))
        if extension:
            extra, cols = extension_features(evaluation, ix, '2025-01-27', recipe['feature_config'], extension)
            x = np.concatenate([x, extra], axis=1)
            names += cols
        assert names == recipe['feature_names']
        fresh = neural(path, x)
        original = np.load(path/'evaluation_predictions.npy')
        error = float(np.max(np.abs(fresh-original)))
        assert error < 2e-5, error
        parts.append(original)
        weights.append(row['weight'])
        errors.append(error)
    composed = blend(parts, weights, manifest['selection']['kind']) if manifest['models'] else parts[0]
    composition_error = float(np.max(np.abs(composed-expected)))
    assert composition_error < 1e-12
    acceptance = dict(passed=True, rows=manifest['rows'], csv_sha256=sha(FINAL/'submission.csv'), manifest_sha256=sha(FINAL/'manifest.json'),
                      csv_rounding_max_error=rounding_error, full_raw_input_new_model_replay_max_error=max(errors, default=0.),
                      composition_max_error=composition_error, previous_best_preserved=True, official_evaluation_labels_available=False)
    write_json(FINAL/'acceptance.json', acceptance)
    print(json.dumps(acceptance, indent=2), flush=True)


def promote():
    manifest = json.loads((FINAL/'manifest.json').read_text())
    accepted = json.loads((FINAL/'acceptance.json').read_text())
    assert accepted['passed'] and accepted['manifest_sha256'] == sha(FINAL/'manifest.json')
    assert accepted['csv_sha256'] == sha(FINAL/'submission.csv')
    selection = json.loads((FINAL/'selection.json').read_text())
    if selection['eligible']:
        write_json(ROOT/'competition/CURRENT_BEST.json', dict(csv=manifest['csv'], sha256=manifest['sha256'],
                   manifest='competition/final_v4/manifest.json', acceptance='competition/final_v4/acceptance.json',
                   validation_26=manifest['validation_26'], validation_25=manifest['validation_25'],
                   previous_best_csv='competition/final_v3/submission.csv', scope='Best validated local recipe; official evaluation labels unavailable.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['export', 'verify', 'promote'])
    a = p.parse_args()
    {'export': export, 'verify': verify, 'promote': promote}[a.command]()
