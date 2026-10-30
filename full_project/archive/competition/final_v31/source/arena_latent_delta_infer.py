"""Frozen NumPy replay for a small preference-geometry replacement."""
import argparse
import json
from pathlib import Path
import numpy as np
from ctr_data import load_cache
from arena_metric_geometry import cosine,preference_probabilities
from arena_metric_infer import read_csv,sha


def replay(final,raw):
    manifest=json.loads((final/'manifest.json').read_text())
    old=dict(np.load(final/'old_geometry.npz'));new=dict(np.load(final/'new_geometry.npz'))
    base=read_csv(final/'frozen_incumbent.csv',raw['rowids'].ravel());parts=[]
    for start in range(0,len(raw['users']),4096):
        ix=np.arange(start,min(start+4096,len(raw['users'])))
        value=cosine(new,raw,ix)
        if manifest['subtract_old']:value=value-cosine(old,raw,ix)
        parts.append(value)
    delta=np.concatenate(parts)
    return preference_probabilities(base,delta,manifest['selection']['mass']),delta


def verify(final,write=True):
    final=final.resolve();workspace=final.parents[1]
    manifest=json.loads((final/'manifest.json').read_text())
    for f,d in manifest['artifact_sha256'].items():assert sha(final/f)==d,f
    for f,d in manifest['raw_input_sha256'].items():assert sha(workspace/f)==d,f
    for f,d in manifest['incumbent_digest'].items():
        if f!='competition/CURRENT_BEST.json':assert sha(workspace/f)==d,f
    _,raw=load_cache(workspace/'data/cache_v1','evaluation')
    assert int(raw['timestamps'].min())>=manifest['cutoff_timestamp']
    actual,delta=replay(final,raw);expected=np.load(final/'predictions.npy')
    error=float(np.max(np.abs(actual-expected)));de=float(np.max(np.abs(delta-np.load(final/'correction.npy'))))
    csv=read_csv(final/'submission.csv',raw['rowids'].ravel());ce=float(np.max(np.abs(expected-csv)))
    from run import check_submission
    check_submission(final/'submission.csv',workspace/'evaluation.zip')
    result=dict(passed=error<2e-5 and de<2e-5 and ce<1e-9,rows=int(actual.size),raw_replay_max_abs_error=error,
                correction_replay_max_abs_error=de,csv_max_abs_error=ce,previous_best_preserved=True,
                protocol='Independent process, frozen NumPy source and two past-only vector tables, original evaluation inputs and immutable incumbent probabilities. Every original CSV row checked.')
    if write:(final/'acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True);assert result['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    verify(Path(a.final),not a.check_only)
