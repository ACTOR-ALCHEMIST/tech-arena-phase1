"""Replay the event-mass correction from frozen raw-input feature assets."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from arena_metric_infer import read_csv,sha


def replay(final,raw):
    from arena_history_data import features,load_first
    from arena_tabular import predict
    from arena_event_mass import features as event_features,correction,apply
    os.environ['ARENA_SNAPSHOT_ROOT']=str(final/'representations')
    recipe=json.loads((final/'first_recipe.json').read_text());cal=dict(np.load(final/'calibrator.npz'))
    torch.set_num_threads(2);device=torch.device('cpu');net=load_first(final/'first_model',device)
    pieces=[]
    for start in range(0,len(raw['users']),2048):
        ix=np.arange(start,min(start+2048,len(raw['users'])))
        x,_=features(raw,ix,'2025-01-27',recipe['feature_config']);p=predict(net,x,device)
        pieces.append(correction(event_features(x,p),cal))
    delta=np.concatenate(pieces)
    base=read_csv(final/'frozen_incumbent.csv',raw['rowids'].ravel())
    selected=json.loads((final/'manifest.json').read_text())['selection']
    return apply(base,delta,selected['mass']),delta


def verify(final,write=True):
    from ctr_data import load_cache
    from run import check_submission
    final=final.resolve();workspace=final.parents[1];m=json.loads((final/'manifest.json').read_text())
    for f,d in m['artifact_sha256'].items():assert sha(final/f)==d,f
    for f,d in m['raw_input_sha256'].items():assert sha(workspace/f)==d,f
    for f,d in m['incumbent_digest'].items():
        if f!='competition/CURRENT_BEST.json':assert sha(workspace/f)==d,f
    _,raw=load_cache(workspace/'data/cache_v1','evaluation');assert int(raw['timestamps'].min())>=m['cutoff_timestamp']
    p,d=replay(final,raw);expected=np.load(final/'predictions.npy');delta=np.load(final/'correction.npy')
    pe=float(np.max(np.abs(p-expected)));de=float(np.max(np.abs(d-delta)))
    csv=read_csv(final/'submission.csv',raw['rowids'].ravel());ce=float(np.max(np.abs(csv-expected)))
    assert sha(final/'submission.csv')==m['sha256'];check_submission(final/'submission.csv',workspace/'evaluation.zip')
    result=dict(passed=pe<2e-6 and de<2e-6 and ce<1e-9,rows=int(p.size),raw_replay_max_abs_error=pe,
                correction_replay_max_abs_error=de,csv_max_abs_error=ce,previous_best_preserved=True,
                protocol='Independent CPU process; frozen first-stage checkpoint, feature source, past-only geometry and event calibrator. All original row IDs and probabilities checked.')
    if write:(final/'acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True);assert result['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    verify(Path(a.final),not a.check_only)
