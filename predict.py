#!/usr/bin/env python3
"""Run frozen V31 inference, verify it, and export the byte-identical CSV."""
import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

# Set before importing numerical libraries; CPU is the portable reference device.
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '6'
ROOT = Path(__file__).resolve().parent
HEADER = ['rowid', 'predicted_probability']
INPUT_HEADER = ['rowid', 'event_id', 'user_id', 'item_id', 'timestamp', 'time_bucket',
                'day_type', 'history_item_ids', 'history_timestamps']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_csv(path, expected_sha, expected_rows):
    require(sha(path) == expected_sha, 'CSV SHA-256 differs from the accepted V31 release')
    seen = set()
    with path.open(encoding='utf-8', newline='') as handle:
        rows = csv.reader(handle)
        require(next(rows) == HEADER, 'Wrong submission header')
        for row in rows:
            require(len(row) == 2 and row[0] not in seen, 'Duplicate rowid or malformed row')
            p = float(row[1])
            require(math.isfinite(p) and 0 <= p <= 1, 'Invalid probability')
            seen.add(row[0])
    require(len(seen) == expected_rows, 'Wrong number of submission rows')


def open_evaluation(path):
    """The caller owns the returned text stream; closing it releases the input."""
    if path.suffix.lower() == '.zip':
        archive = zipfile.ZipFile(path)
        names = [n for n in archive.namelist() if n.lower().endswith('.csv')]
        require(len(names) == 1, 'Evaluation ZIP must contain exactly one CSV')
        # ZipExtFile retains the shared archive handle until the stream closes.
        stream = io.TextIOWrapper(archive.open(names[0]), encoding='utf-8', newline='')
        archive.close()
        return stream
    return path.open(encoding='utf-8', newline='')


def read_evaluation(path, manifest):
    import numpy as np
    expected = manifest['evaluation_zip_sha256' if path.suffix.lower() == '.zip' else 'evaluation_csv_sha256']
    require(sha(path) == expected, 'This minimal package supports only the original competition evaluation input')
    with gzip.open(ROOT/'assets/vocabulary.json.gz', 'rt', encoding='utf-8') as handle:
        vocabulary = json.load(handle)
    nu, ni = vocabulary['users'], vocabulary['items']
    n = manifest['events']
    data = dict(users=np.zeros(n, 'int32'), items=np.zeros((n, 10), 'int32'),
                timestamps=np.zeros(n, 'int64'), history=np.zeros((n, 20), 'int32'),
                age_hours=np.zeros((n, 20), 'float32'), context=np.zeros((n, 6), 'float32'),
                rowids=np.empty((n, 10), 'S64'))
    count = 0
    with open_evaluation(path) as handle:
        reader = csv.reader(handle)
        require(next(reader) == INPUT_HEADER, 'Wrong evaluation header')
        for eid, group in itertools.groupby(reader, key=lambda r: r[1]):
            rows = list(group)
            require(count < n and len(rows) == 10, 'Each event must have 10 candidates')
            require(all(r[2:3]+r[4:9] == rows[0][2:3]+rows[0][4:9] for r in rows),
                    'Inconsistent event context: '+eid)
            row, index = rows[0], count
            ts, hids, hts = int(row[4]), json.loads(row[7]), json.loads(row[8])
            require(len(hids) == len(hts) and len(hids) <= 20, 'Invalid history length')
            require(all(t < ts for t in hts), 'History must precede the event')
            data['users'][index] = nu[row[2]]
            data['items'][index] = [ni[r[3]] for r in rows]
            data['timestamps'][index] = ts
            if hids:
                data['history'][index, :len(hids)] = [ni[h] for h in hids]
                data['age_hours'][index, :len(hids)] = [(ts-t)/3600 for t in hts]
            hour = (ts % 86400)/3600
            gap = (ts-hts[-1])/3600 if hts else 0
            span = (hts[-1]-hts[0])/3600 if hts else 0
            data['context'][index] = [math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24),
                                      float(row[6]), len(hids)/20,
                                      math.log1p(gap)/math.log1p(24*30), math.log1p(span)/math.log1p(24*30)]
            data['rowids'][index] = [r[0].encode('utf-8') for r in rows]
            count += 1
    require(count == n, 'Wrong evaluation event count')
    return data


def infer(evaluation, manifest):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from features import build_features
    from models import FreshMF, WideSlate
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    print('Reading the original evaluation input...', flush=True)
    raw = read_evaluation(evaluation, manifest)
    weights = torch.load(ROOT/'assets/weights.pt', map_location='cpu', weights_only=True)
    fresh = {epoch: FreshMF(state) for epoch, state in weights['fresh'].items()}
    print('Building 323 frozen-representation features...', flush=True)
    with threadpool_limits(limits=1):
        x, names = build_features(raw, ROOT/'assets', fresh)
        require(names == manifest['feature_columns'], 'Feature column order differs from V31')
        print('Running the V31 neural head on CPU...', flush=True)
        prediction = WideSlate(weights['head']).predict(x)
    require(np.isfinite(prediction).all(), 'Non-finite model output')
    with np.load(ROOT/'assets/ensemble.npz', allow_pickle=False) as saved:
        baseline, reference_head = saved['baseline'], saved['head_reference']
    error = float(np.max(np.abs(prediction-reference_head)))
    require(error <= manifest['head_tolerance'], f'Neural inference differs from the release: {error}')
    logit = lambda p: np.log(np.clip(p, 1e-7, 1-1e-7)/(1-np.clip(p, 1e-7, 1-1e-7)))
    z = .95*logit(baseline)+.05*logit(prediction)
    probabilities = 1/(1+np.exp(-np.clip(z, -40, 40)))
    with gzip.open(ROOT/'assets/reference.csv.gz', 'rt', encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        require(next(reader) == HEADER, 'Wrong reference header')
        rows = list(reader)
    ids = np.array([r[0].encode('utf-8') for r in rows], dtype='S64')
    require(np.array_equal(ids, raw['rowids'].ravel()), 'Evaluation rowids/order differ from the release')
    reference = np.array([float(r[1]) for r in rows]).reshape(-1, 10)
    final_error = float(np.max(np.abs(probabilities-reference)))
    require(final_error <= manifest['blend_tolerance'], f'Blended probabilities differ from V31: {final_error}')
    report = dict(mode='v31_neural_inference_with_frozen_v30', device='cpu',
                  feature_width=len(names), head_max_absolute_error=error,
                  blend_max_absolute_error=final_error, head_tolerance=manifest['head_tolerance'],
                  blend_tolerance=manifest['blend_tolerance'],
                  output_policy='Verify fresh inference, then export canonical frozen CSV bytes for exact reproducibility')
    return report, raw, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=ROOT/'assets/evaluation.zip')
    parser.add_argument('--out', type=Path, default=ROOT/'submission.csv')
    parser.add_argument('--replay', action='store_true', help='Export the frozen CSV only; no model inference or third-party dependencies')
    parser.add_argument('--numerical-output', action='store_true', help='Write fresh numerical predictions instead of canonical bytes; exact SHA is not guaranteed')
    args = parser.parse_args()
    require(not (args.replay and args.numerical_output), '--replay cannot be combined with --numerical-output')
    out = args.out.resolve()
    protected = [ROOT/'assets', ROOT/'predict.py', ROOT/'models.py', ROOT/'features.py',
                 ROOT/'manifest.json', ROOT/'README.md', ROOT/'requirements.txt', ROOT/'VERIFICATION.json']
    require(not any(out == p or p in out.parents for p in protected), 'Output would overwrite a package input')
    manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    started = time.monotonic()
    for relative, expected in manifest['asset_sha256'].items():
        require(sha(ROOT/relative) == expected, 'Asset SHA-256 mismatch: '+relative)
    if args.replay:
        report, raw, probabilities = dict(mode='frozen_csv_export', model_inference_performed=False), None, None
    else:
        report, raw, probabilities = infer(args.evaluation.resolve(), manifest)
        report['model_inference_performed'] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, prefix='.v31-', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if args.numerical_output:
            with temporary.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle, lineterminator='\r\n')
                writer.writerow(HEADER)
                writer.writerows((rid.decode(), f'{float(p):.10g}')
                                 for rid, p in zip(raw['rowids'].ravel(), probabilities.ravel()))
        else:
            with gzip.open(ROOT/'assets/reference.csv.gz', 'rb') as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target)
            validate_csv(temporary, manifest['expected_csv_sha256'], manifest['rows'])
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha(out)
    report.update(passed=True, rows=manifest['rows'], csv_sha256=digest,
                  byte_identical_to_v31=digest == manifest['expected_csv_sha256'],
                  canonical_output=not args.numerical_output, seconds=round(time.monotonic()-started, 3),
                  python=sys.version.split()[0], official_evaluation_labels_available=False)
    if not args.replay:
        from importlib.metadata import version
        report['packages'] = {name: version(name) for name in ['numpy', 'scipy', 'torch', 'implicit', 'numba', 'threadpoolctl']}
    report_path = out.with_name(out.name+'.verification.json')
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print('CSV: '+str(out), flush=True)


if __name__ == '__main__':
    main()
