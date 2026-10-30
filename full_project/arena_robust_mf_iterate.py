"""Second campaign: online corruption and supervised embedding fine tuning."""
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from arena_robust_mf import ROOT, OUT, RECIPES
from arena_ood import sha, write_json, guard


def command(args, label):
    write_json(OUT/'state.json', dict(stage=label, pid=os.getpid(), started=time.time(), command=args))
    print(json.dumps({'stage': label, 'command': args}), flush=True)
    with (OUT/(label+'.log')).open('a') as log:
        subprocess.run([sys.executable, '-u']+args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def record(name, cutoff):
    path = OUT/'runs'/name/cutoff
    return json.loads((path/'result.json').read_text()), np.load(path/'validation.npz')


def select():
    from arena_finalize import blend
    from arena_validation import importance_weights, weighted_auc
    from sklearn.metrics import roc_auc_score
    incumbent = json.loads((OUT/'incumbent_pointer.json').read_text())
    parent = (ROOT/incumbent['csv']).parent
    names = list(RECIPES)
    groups = {}
    for cutoff in ['2025-01-26', '2025-01-25']:
        baseline = np.load(parent/('validation_'+cutoff[-2:]+'.npz'))
        parts = [baseline['predictions']]
        for name in names:
            _, d = record(name, cutoff)
            for key in ['event_indices', 'labels']:
                np.testing.assert_array_equal(d[key], baseline[key])
            parts.append(d['predictions'])
        iw, _ = importance_weights(cutoff, baseline['event_indices'])
        groups[cutoff] = (baseline, parts, iw)
    weights = [[1.]+[0.]*len(names)]
    for j in range(len(names)):
        for mass in [.05, .10, .25, .5, .75, 1.]:
            w = [1-mass]+[0.]*len(names)
            w[j+1] = mass
            weights.append(w)
    for mass in [.5, 1.]:
        weights.append([1-mass]+[mass/len(names)]*len(names))
    scores = []
    for w in weights:
        row = dict(weights=w, kind='logit')
        for cutoff, (d, parts, iw) in groups.items():
            p = blend(parts, w, 'logit')
            row['auc_'+cutoff[-2:]] = float(roc_auc_score(d['labels'].ravel(), p.ravel()))
            row['weighted_auc_'+cutoff[-2:]] = weighted_auc(d['labels'], p, iw)
        row['selection_score'] = .5*(row['weighted_auc_26']+row['weighted_auc_25'])
        scores.append(row)
    old = scores[0]
    eligible = [r for r in scores[1:] if all(r[k] > old[k]+1e-6 for k in ['auc_26', 'auc_25', 'selection_score'])]
    winner = max(eligible, key=lambda r: r['selection_score']) if eligible else old
    result = dict(sources=['frozen_incumbent']+names, incumbent=old, selected=winner, eligible=len(eligible), recipes=scores,
                  candidates=[dict(name=n, primary=record(n, '2025-01-26')[0], secondary=record(n, '2025-01-25')[0]) for n in names])
    write_json(OUT/'selection.json', result)
    for cutoff, (d, parts, _) in groups.items():
        np.savez_compressed(OUT/('selected_validation_'+cutoff[-2:]+'.npz'), event_indices=d['event_indices'], labels=d['labels'], predictions=blend(parts, winner['weights'], 'logit'))
    print(json.dumps({'stage': 'selection', 'selected': winner, 'incumbent': old, 'eligible': len(eligible)}), flush=True)
    return result


def export_and_verify():
    import torch
    from ctr_data import load_cache
    from arena_finalize import blend, export_csv
    from arena_robust_mf import load_model, predict
    from arena_ood_export import csv_probabilities
    from run import check_submission
    selection = json.loads((OUT/'selection.json').read_text())
    if not selection['eligible']:
        return None
    final = ROOT/'competition/final_v5'
    final.mkdir(exist_ok=True)
    before = guard()
    pointer = json.loads((OUT/'incumbent_pointer.json').read_text())
    _, data = load_cache(ROOT/'data/cache_v1', 'evaluation')
    shutil.copy2(ROOT/pointer['csv'], final/'frozen_incumbent.csv')
    parts = [csv_probabilities(final/'frozen_incumbent.csv', data['rowids'].ravel())]
    weights = [selection['selected']['weights'][0]]
    models = []
    for name, weight in zip(selection['sources'][1:], selection['selected']['weights'][1:]):
        if weight > 0:
            r, _ = record(name, '2025-01-26')
            command(['arena_robust_mf.py', '--name', name, '--cutoff', '2025-01-27', '--fixed-epochs', str(r['best_epoch']), '--final'], name+'_final')
            path = OUT/'runs'/name/'2025-01-27'
            copied = final/'models'/name
            shutil.copytree(path, copied, dirs_exist_ok=True)
            parts.append(np.load(copied/'evaluation_predictions.npy'))
            weights.append(weight)
            models.append(dict(name=name, weight=weight, path=str(copied.relative_to(ROOT)), model_sha256=sha(copied/'model.pt')))
    p = blend(parts, weights, 'logit')
    export_csv(final/'submission.csv', data, p)
    np.save(final/'predictions.npy', p)
    shutil.copy2(OUT/'selection.json', final/'selection.json')
    for c in ['26', '25']:
        shutil.copy2(OUT/('selected_validation_'+c+'.npz'), final/('validation_'+c+'.npz'))
    # Reload weights, then replay all official inputs with no embedding caches.
    errors = []
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(2)
    for row in models:
        path = ROOT/row['path']
        fresh = predict(load_model(path, device), data, np.arange(len(data['users'])), device)
        expected = np.load(path/'evaluation_predictions.npy')
        error = float(np.max(np.abs(fresh-expected)))
        assert error < 2e-5
        errors.append(error)
    check_submission(final/'submission.csv', ROOT/'evaluation.zip')
    written = csv_probabilities(final/'submission.csv', data['rowids'].ravel())
    assert np.max(np.abs(written-p)) < 1e-9
    source = final/'source'
    source.mkdir(exist_ok=True)
    for file in ['arena_robust_mf.py', 'arena_robust_mf_iterate.py', 'test_arena_robust_mf.py', 'arena_neural_v2.py', 'arena_ood.py', 'arena_core.py',
                 'arena_tabular.py', 'arena_finalize.py', 'arena_ood_export.py', 'ctr_data.py', 'run.py', 'requirements-v3.txt']:
        shutil.copy2(ROOT/file, source/file)
    manifest = dict(method='v5_supervised_multiview_mf_online_corruption', selection=selection['selected'], models=models,
                    previous_best=pointer, incumbent_digest=before, baseline_weight=weights[0], rows=len(p)*10,
                    csv='competition/final_v5/submission.csv', sha256=sha(final/'submission.csv'),
                    validation_26={'auc': selection['selected']['auc_26']}, validation_25={'auc': selection['selected']['auc_25']},
                    artifact_sha256={str(f.relative_to(final)):sha(f) for f in final.rglob('*') if f.is_file() and f.name not in ['manifest.json','acceptance.json'] and not f.name.endswith('.lock')},
                    raw_input_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'data/cache_v1/evaluation').glob('*.npy')},
                    official_evaluation_labels_available=False)
    write_json(final/'manifest.json', manifest)
    write_json(final/'acceptance.json', dict(passed=True, csv_sha256=manifest['sha256'], manifest_sha256=sha(final/'manifest.json'), rows=len(p)*10,
               checkpoint_full_input_replay_max_error=max(errors, default=0.), csv_rounding_max_error=float(np.max(np.abs(written-p))), incumbent_unchanged=guard()==before))
    assert guard() == before
    command(['arena_robust_mf_iterate.py', '--verify'], 'final_fresh_verify')
    write_json(ROOT/'competition/CURRENT_BEST.json', dict(csv=manifest['csv'], sha256=manifest['sha256'], manifest='competition/final_v5/manifest.json',
               acceptance='competition/final_v5/acceptance.json', validation_26=manifest['validation_26'], validation_25=manifest['validation_25'],
               previous_best_csv=pointer['csv'], scope='Best validated local recipe; official evaluation labels unavailable.'))
    return manifest


def verify_final():
    import torch
    from ctr_data import load_cache
    from arena_finalize import blend
    from arena_robust_mf import load_model, predict
    from arena_ood_export import csv_probabilities
    from run import check_submission
    final = ROOT/'competition/final_v5'
    manifest = json.loads((final/'manifest.json').read_text())
    for file, digest in manifest['artifact_sha256'].items():
        assert sha(final/file) == digest, file
    for file, digest in manifest['raw_input_sha256'].items():
        assert sha(ROOT/file) == digest, file
    for file, digest in manifest['incumbent_digest'].items():
        if file != 'competition/CURRENT_BEST.json':
            assert sha(ROOT/file) == digest, file
    check_submission(final/'submission.csv', ROOT/'evaluation.zip')
    _, data = load_cache(ROOT/'data/cache_v1', 'evaluation')
    parts = [csv_probabilities(final/'frozen_incumbent.csv', data['rowids'].ravel())]
    weights, errors = [manifest['baseline_weight']], []
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(2)
    for row in manifest['models']:
        path = ROOT/row['path']
        fresh = predict(load_model(path, device), data, np.arange(len(data['users'])), device)
        expected = np.load(path/'evaluation_predictions.npy')
        error = float(np.max(np.abs(fresh-expected)))
        assert error < 2e-5
        errors.append(error)
        parts.append(expected)
        weights.append(row['weight'])
    expected = np.load(final/'predictions.npy')
    composed = blend(parts, weights, 'logit')
    composition_error = float(np.max(np.abs(composed-expected)))
    assert composition_error < 1e-12
    written = csv_probabilities(final/'submission.csv', data['rowids'].ravel())
    rounding = float(np.max(np.abs(written-expected)))
    assert rounding < 1e-9
    accepted = dict(passed=True, rows=len(expected)*10, csv_sha256=sha(final/'submission.csv'), manifest_sha256=sha(final/'manifest.json'),
                    fresh_process_full_input_replay_max_error=max(errors, default=0.), csv_rounding_max_error=rounding,
                    composition_max_error=composition_error, previous_best_preserved=True)
    write_json(final/'acceptance.json', accepted)
    print(json.dumps(accepted, indent=2), flush=True)


def report():
    selection = json.loads((OUT/'selection.json').read_text())
    winner, old = selection['selected'], selection['incumbent']
    lines = ['# 在线历史增强与曝光监督 embedding 微调', '',
             '在 V4 后比较两种协议：先用完整拟合期预训练、再在同批曝光训练评分头；以及严格用更早前缀预训练、评分头只训练之后的新曝光（causal）。后者避免评分头依赖已经记住自身点击的用户表示。比较冻结 embedding 与小学习率微调。', '',
             '每个 minibatch 动态采样历史删除，按事件共享 10 个候选的历史，更新长度／间隔／时间跨度及所有历史匹配值。另对 5% 用户隐藏 ID，历史保留。', '',
             '| 配置 | Jan26 AUC | Jan25 AUC | 训练事件数 | 主验证选中轮数 |', '|---|---:|---:|---:|---:|']
    for row in selection['candidates']:
        a, b = row['primary'], row['secondary']
        lines.append(f"| {row['name']} | {a['auc']:.9f} | {b['auc']:.9f} | {a['training_events']:,} | {a['best_epoch']} |")
    lines += ['', f"此前最佳：Jan26 {old['auc_26']:.9f}，Jan25 {old['auc_25']:.9f}。本轮选中配方：Jan26 {winner['auc_26']:.9f}，Jan25 {winner['auc_25']:.9f}。", '',
              '两切点原始 AUC 与平均群体加权 AUC 均需提高才晋升；官方评测标签不可用。', '',
              '已验证候选顺序等变性、空历史输出、在线全删历史后的上下文一致性、embedding 梯度和 padding 梯度。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    readme = ROOT/'README.md'
    body = readme.read_text()
    pointer = json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    block = ['<!-- ROBUST_MF_START -->', '# 继续冲分：曝光监督与在线增强', '',
             f"新增 {len(selection['candidates'])} 组曝光监督／前缀训练配置及双切点复核。当前最优 Jan26 AUC **{pointer['validation_26']['auc']:.9f}**，Jan25 **{pointer['validation_25']['auc']:.9f}**；0.835 {'已达到（本地主验证）' if pointer['validation_26']['auc'] >= .835 else '尚未达到'}。", '',
             f"[当前最优 CSV]({pointer['csv']}) · [本轮报告](competition/robust_mf/REPORT.md)", '', '<!-- ROBUST_MF_END -->', '']
    if '<!-- ROBUST_MF_START -->' in body:
        body = body.split('<!-- ROBUST_MF_END -->', 1)[1].lstrip('\n')
    readme.write_text('\n'.join(block)+'\n'+body)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.iteration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        write_json(OUT/'state.json', dict(stage='waiting_ood_delivery_lock', pid=os.getpid()))
        with (ROOT/'competition/ood/.delivery.lock').open('a') as previous:
            fcntl.flock(previous, fcntl.LOCK_EX)
        accepted = json.loads((ROOT/'competition/ood/delivery_state.json').read_text())
        assert accepted['stage'] == 'delivery_completed', 'Finish V4 before starting this campaign'
        before = guard()
        if (OUT/'incumbent.json').exists():
            assert json.loads((OUT/'incumbent.json').read_text()) == before
        else:
            write_json(OUT/'incumbent.json', before)
            shutil.copy2(ROOT/'competition/CURRENT_BEST.json', OUT/'incumbent_pointer.json')
        for name in RECIPES:
            command(['arena_robust_mf.py', '--name', name], name+'_primary')
            r, _ = record(name, '2025-01-26')
            command(['arena_robust_mf.py', '--name', name, '--cutoff', '2025-01-25', '--fixed-epochs', str(r['best_epoch'])], name+'_locked25')
        assert guard() == before
        select()
        export_and_verify()
        report()
        write_json(OUT/'state.json', dict(stage='completed', pid=os.getpid(), finished=time.time()))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    verify_final() if a.verify else run()
