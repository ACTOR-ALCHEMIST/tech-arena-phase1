"""Freeze, replay, validate and promote the new campaign's best local recipe."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from arena_ood import ROOT, guard, sha, write_json
from arena_push084 import OUT

FINAL = ROOT/'competition/final_v6'
TARGET_AUC = .84


def representation_ignore(directory,names):
    """Copy complete inference assets, excluding mutable optimizer/temp files."""
    ignored=[]
    for name in names:
        path=Path(directory)/name
        if name.endswith('.lock') or '.tmp.' in name or name=='resume.pt':ignored.append(name)
        elif path.is_dir() and name.startswith('spherical_') and not (path/'result.json').exists():
            # Unrelated representation experiments may still be fitting in the
            # shared snapshot. Required factories finished before prediction.
            ignored.append(name)
    return ignored


def export(verify_script='arena_push084_export.py', extra_sources=None):
    from arena_finalize import blend, export_csv
    from ctr_data import load_cache
    from arena_ood_export import csv_probabilities
    import arena_ood_export as tabular_export
    selection = json.loads((OUT/'selection.json').read_text())
    if not selection['eligible']:
        print('No candidate improved both chronological splits; incumbent retained.', flush=True)
        return
    anchor_file = ROOT/'competition/final_v6/audit_24.json'
    audit_file = OUT/'audit_24.json'
    if anchor_file.exists() and audit_file.exists():
        anchor, candidate = json.loads(anchor_file.read_text()), json.loads(audit_file.read_text())
        checked = dict(overall_floor=anchor['new_auc']-.0001,
                       weekday_floor=anchor['cohorts']['weekday']['new_auc']-.0001,
                       candidate_overall=candidate['new_auc'], candidate_weekday=candidate['cohorts']['weekday']['new_auc'])
        checked['passed'] = checked['candidate_overall'] >= checked['overall_floor'] and checked['candidate_weekday'] >= checked['weekday_floor']
        write_json(OUT/'export_ood_floor.json', checked)
        if not checked['passed']:
            raise RuntimeError('Candidate failed the absolute V6 OOD floor; incumbent preserved. Resume with the updated campaign controller to record rejection.')
    if (FINAL/'manifest.json').exists():
        raise RuntimeError('V6 already frozen; use a new version for subsequent campaigns')
    before = guard()
    assert before == json.loads((OUT/'incumbent.json').read_text())
    pointer = json.loads((OUT/'incumbent_pointer.json').read_text())
    FINAL.mkdir(parents=True, exist_ok=True)
    _, evaluation = load_cache(ROOT/'data/cache_v1', 'evaluation')
    shutil.copy2(ROOT/pointer['csv'], FINAL/'frozen_incumbent.csv')
    parts = [csv_probabilities(FINAL/'frozen_incumbent.csv', evaluation['rowids'].ravel())]
    weights = [selection['selected']['weights'][0]]
    models = []
    tabular_export.FINAL = FINAL
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(2)
    for name, weight in zip(selection['sources'][1:], selection['selected']['weights'][1:]):
        if weight <= 0:
            continue
        if name == 'refreshed_mf':
            from arena_robust_mf import load_model, refresh_model, predict
            original = ROOT/'competition/final_v5/models/causal_frozen'
            model = refresh_model(load_model(original, device), '2025-01-27', device)
            path = FINAL/'models'/name
            path.mkdir(parents=True, exist_ok=True)
            meta, _ = load_cache(ROOT/'data/cache_v1', 'train')
            torch.save(dict(state_dict=model.state_dict(), config=model.config, sizes=model.sizes, cache_id=meta['cache_id']), path/'model.pt')
            p = predict(model, evaluation, np.arange(len(evaluation['users'])), device)
            np.save(path/'evaluation_predictions.npy', p)
            write_json(path/'recipe.json', dict(name=name, frozen_head_source=str(original.relative_to(ROOT)),
                       head_source_sha256=sha(original/'model.pt'), inference_snapshot='2025-01-27'))
            kind = 'native'
        else:
            path = tabular_export.fit(name)
            recipe = json.loads((path/'recipe.json').read_text())
            kind = 'bayes_mf' if recipe.get('bayes_mf') else 'native_attention' if recipe.get('native_attention') else 'graph_tree' if recipe.get('graph_tree') else 'tabular'
            p = np.load(path/'evaluation_predictions.npy')
        parts.append(p)
        weights.append(weight)
        model_file = recipe.get('model_file', 'model.cbm') if kind == 'graph_tree' else 'model.pt'
        models.append(dict(name=name, weight=weight, path=str(path.relative_to(ROOT)), kind=kind, model_file=model_file, model_sha256=sha(path/model_file)))
    p = blend(parts, weights, selection['selected']['kind'])
    export_csv(FINAL/'submission.csv', evaluation, p)
    np.save(FINAL/'predictions.npy', p)
    for file in ['selection.json', 'selected_validation_26.npz', 'selected_validation_25.npz']:
        target = file.replace('selected_validation_', 'validation_')
        shutil.copy2(OUT/file, FINAL/target)
    for day in ['26', '25', '24']:
        path = OUT/('audit_'+day+'.json')
        if path.exists():
            assert json.loads(path.read_text())['selection_sha256'] == sha(OUT/'selection.json'), 'Recompute audit for the frozen final recipe'
            shutil.copy2(path, FINAL/path.name)
    if (OUT/'selected_validation_24.npz').exists():
        shutil.copy2(OUT/'selected_validation_24.npz', FINAL/'validation_24.npz')
    for file in ['shift_profile.json', 'normalizer_audit.json', 'validation_overlap.json', 'static_selection.json', 'stress.json', 'tests.log']:
        if (OUT/file).exists():
            shutil.copy2(OUT/file, FINAL/file)
    if any(r['kind'] in ['tabular', 'native_attention', 'graph_tree'] for r in models):
        from arena_core import snapshot
        shutil.copytree(snapshot('2025-01-27'), FINAL/'representations/snapshot_2025-01-27', dirs_exist_ok=True,ignore=representation_ignore)
    source = FINAL/'source'
    source.mkdir(exist_ok=True)
    files = ['arena_push084.py', 'arena_push084_export.py', 'arena_push084_audit.py', 'arena_push084_report.py', 'arena_push084_prepare.py', 'arena_push084_reblend.py', 'arena_push084_stress.py', 'arena_coclick.py', 'arena_exposure_repr.py', 'arena_spectral.py', 'arena_softmax_repr.py', 'arena_joint_graph.py', 'arena_bias_repr.py', 'arena_weekend.py', 'arena_daily.py', 'arena_origins.py',
             'arena_objectives.py', 'arena_push_model.py', 'test_arena_coclick.py', 'test_arena_push084.py',
             'arena_ood.py', 'arena_ood_features.py', 'arena_ood_consensus.py', 'arena_ood_context.py', 'arena_ood_audit.py', 'arena_ood_export.py',
             'arena_robust_mf.py', 'arena_core.py', 'arena_features.py', 'arena_sgns_features.py', 'arena_sgns.py', 'arena_latent.py',
             'arena_feedback.py', 'arena_tabular.py', 'arena_neural_v2.py', 'arena_finalize.py', 'arena_models.py', 'arena_report.py',
             'ctr_data.py', 'run.py', 'requirements-v3.txt']
    files += ['arena_windows.py', 'test_arena_campaign_families.py']+(extra_sources or [])
    for file in files:
        shutil.copy2(ROOT/file, source/file)
    manifest = dict(method=FINAL.name+'_past_only_multiview_temporal_ensemble', selection=selection['selected'],
                    sources=selection['sources'], models=models, baseline_weight=weights[0], previous_best=pointer,
                    incumbent_digest=before, rows=len(p)*10, csv=str((FINAL/'submission.csv').relative_to(ROOT)), sha256=sha(FINAL/'submission.csv'),
                    validation_26=dict(auc=selection['selected']['auc_26']), validation_25=dict(auc=selection['selected']['auc_25']),
                    artifact_sha256={str(f.relative_to(FINAL)): sha(f) for f in FINAL.rglob('*') if f.is_file() and f.name not in ['manifest.json', 'acceptance.json'] and not f.name.endswith('.lock')},
                    raw_input_sha256={str(f.relative_to(ROOT)): sha(f) for f in (ROOT/'data/cache_v1/evaluation').glob('*.npy')},
                    official_evaluation_labels_available=False, target_auc=TARGET_AUC,
                    target_reached_on_primary_validation=selection['selected']['auc_26'] >= TARGET_AUC)
    if (OUT/'audit_24.json').exists():
        manifest['validation_24'] = dict(auc=json.loads((OUT/'audit_24.json').read_text())['new_auc'])
    assert guard() == before
    write_json(FINAL/'manifest.json', manifest)
    subprocess.run([sys.executable, '-u', verify_script, '--verify'], cwd=ROOT, check=True)
    assert guard() == before
    acceptance = json.loads((FINAL/'acceptance.json').read_text())
    assert acceptance['passed']
    write_json(ROOT/'competition/CURRENT_BEST.json', dict(csv=manifest['csv'], sha256=manifest['sha256'],
               manifest=str((FINAL/'manifest.json').relative_to(ROOT)), acceptance=str((FINAL/'acceptance.json').relative_to(ROOT)),
               validation_26=manifest['validation_26'], validation_25=manifest['validation_25'], previous_best_csv=pointer['csv'],
               validation_24=manifest.get('validation_24'),
               scope='Best validated local recipe; official evaluation labels unavailable.'))
    write_json(OUT/'delivery.json', dict(stage='completed', current_best=manifest['csv'], acceptance=acceptance))


def verify():
    from ctr_data import load_cache
    from arena_finalize import blend
    from arena_ood_export import csv_probabilities
    from arena_ood_audit import neural
    from run import check_submission
    manifest = json.loads((FINAL/'manifest.json').read_text())
    for f, digest in manifest['artifact_sha256'].items():
        assert sha(FINAL/f) == digest, f
    for f, digest in manifest['raw_input_sha256'].items():
        assert sha(ROOT/f) == digest, f
    for f, digest in manifest['incumbent_digest'].items():
        if f != 'competition/CURRENT_BEST.json':
            assert sha(ROOT/f) == digest, f
    check_submission(FINAL/'submission.csv', ROOT/'evaluation.zip')
    _, evaluation = load_cache(ROOT/'data/cache_v1', 'evaluation')
    ix = np.arange(len(evaluation['users']))
    expected = np.load(FINAL/'predictions.npy')
    parts = [csv_probabilities(FINAL/'frozen_incumbent.csv', evaluation['rowids'].ravel())]
    weights, errors = [manifest['baseline_weight']], []
    os.environ['ARENA_SNAPSHOT_ROOT'] = str(FINAL/'representations')
    torch.set_num_threads(2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    for row in manifest['models']:
        path = ROOT/row['path']
        if row['kind'] == 'native':
            from arena_robust_mf import load_model, predict
            fresh = predict(load_model(path, device), evaluation, ix, device)
        elif row['kind'] == 'bayes_mf':
            from arena_bayes_mf import load_model, predict
            fresh = predict(load_model(path, device), evaluation, ix, device)
        else:
            from arena_features import complete_features
            from arena_ood_features import extension_features
            recipe = json.loads((path/'recipe.json').read_text())
            x, names = complete_features(evaluation, ix, '2025-01-27', recipe['feature_config'])
            aug = recipe['augmentation_config']
            extension = aug.get('feature_mode', aug.get('expanded', False))
            if extension:
                extra, cols = extension_features(evaluation, ix, '2025-01-27', recipe['feature_config'], extension)
                x, names = np.concatenate([x, extra], 1), names+cols
            if aug.get('origin_days') or aug.get('origin_start'):
                from arena_origins import delay, DELAY_NAME
                x, names = np.concatenate([x, delay(evaluation, ix, '2025-01-27')], 1), names+[DELAY_NAME]
            if row['kind'] == 'graph_tree':
                from arena_graph_trees import matrix
                x, names = matrix(x, names, recipe['model_config'].get('slate_context'))
            assert names == recipe['feature_names']
            if row['kind'] == 'native_attention':
                from arena_native_attention import load_model, predict
                fresh = predict(load_model(path, device), evaluation, ix, x, device)
            elif row['kind'] == 'graph_tree':
                from arena_graph_trees import predict
                fresh = predict(path, x)
            else:
                fresh = neural(path, x)
            del x
        saved = np.load(path/'evaluation_predictions.npy')
        error = float(np.max(np.abs(fresh-saved)))
        assert error < 2e-5, (row['name'], error)
        errors.append(error)
        parts.append(saved)
        weights.append(row['weight'])
    composed = blend(parts, weights, manifest['selection']['kind'])
    error = float(np.max(np.abs(composed-expected)))
    assert error < 1e-12
    written = csv_probabilities(FINAL/'submission.csv', evaluation['rowids'].ravel())
    rounding = float(np.max(np.abs(written-expected)))
    assert rounding < 1e-9
    accepted = dict(passed=True, rows=len(expected)*10, csv_sha256=sha(FINAL/'submission.csv'), manifest_sha256=sha(FINAL/'manifest.json'),
                    fresh_process_full_raw_input_replay_max_error=max(errors, default=0.), composition_max_error=error,
                    csv_rounding_max_error=rounding, previous_best_preserved=True, official_evaluation_labels_available=False)
    write_json(FINAL/'acceptance.json', accepted)
    print(json.dumps(accepted, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    verify() if a.verify else export()
