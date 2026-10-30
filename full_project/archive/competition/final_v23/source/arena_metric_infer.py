"""Independent frozen-source raw replay for a conditional metric correction."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from ctr_data import load_cache
from arena_metric_geometry import profiles,correction,preference_probabilities


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def read_csv(path,rowids):
    with Path(path).open(newline='') as f:
        reader=csv.reader(f);assert next(reader)==['rowid','predicted_probability']
        values=np.empty(len(rowids),np.float64)
        for n,row in enumerate(reader):
            assert n<len(rowids) and len(row)==2
            assert row[0]==rowids[n].decode('utf-8')
            values[n]=float(row[1])
        assert n+1==len(rowids)
    assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
    return values.reshape(-1,10)


def replay(final,raw):
    rep=dict(np.load(final/'representation.npz'));matrix=np.load(final/'metric.npz')['matrix']
    rowids=raw['rowids'].ravel();base=read_csv(final/'frozen_incumbent.csv',rowids)
    parts=[]
    for start in range(0,len(raw['users']),4096):
        ix=np.arange(start,min(start+4096,len(raw['users'])))
        p,v=profiles(raw,ix,rep);parts.append(correction(p,v,matrix))
    delta=np.concatenate(parts)
    manifest=json.loads((final/'manifest.json').read_text())
    return preference_probabilities(base,delta,manifest['selection']['mass']),delta


def verify(final):
    final=final.resolve();workspace=final.parents[1]
    manifest=json.loads((final/'manifest.json').read_text())
    for f,digest in manifest['artifact_sha256'].items():assert sha(final/f)==digest,f
    for f,digest in manifest['raw_input_sha256'].items():assert sha(workspace/f)==digest,f
    for f,digest in manifest['incumbent_digest'].items():assert sha(workspace/f)==digest,f
    _,raw=load_cache(workspace/'data/cache_v1','evaluation')
    actual,delta=replay(final,raw);expected=np.load(final/'predictions.npy')
    pred_error=float(np.max(np.abs(actual-expected)))
    delta_error=float(np.max(np.abs(delta-np.load(final/'correction.npy'))))
    csv=read_csv(final/'submission.csv',raw['rowids'].ravel())
    csv_error=float(np.max(np.abs(expected-csv)))
    assert sha(final/'submission.csv')==manifest['sha256']
    from run import check_submission
    check_submission(final/'submission.csv',workspace/'evaluation.zip')
    result=dict(passed=pred_error<2e-5 and delta_error<2e-5 and csv_error<1e-9,rows=int(actual.size),
                raw_replay_max_abs_error=pred_error,metric_replay_max_abs_error=delta_error,csv_max_abs_error=csv_error,
                protocol='Separate process, frozen NumPy inference source, frozen past-only geometry and metric, original evaluation input arrays; every CSV row ID verified.')
    (final/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    assert result['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',required=True);a=p.parse_args();verify(Path(a.final))
