"""Label-free cohort shift diagnostics; not an official test score estimate."""
import numpy as np
from sklearn.metrics import roc_auc_score
from ctr_data import load_cache
from run import timestamp


def cohort_groups(data,indices,known_users):
    length=(data["history"][indices]!=0).sum(1)
    bins=np.digitize(length,[1,5,10,20])
    return bins*2+known_users[data["users"][indices]].astype(np.int32)


def importance_weights(cutoff,event_indices):
    meta,train=load_cache("data/cache_v1","train");_,evaluation=load_cache("data/cache_v1","evaluation")
    fit=np.flatnonzero(train["timestamps"]<timestamp(cutoff))
    known=np.zeros(meta["n_users"],bool);known[train["users"][fit]]=True
    final_known=np.zeros(meta["n_users"],bool);final_known[train["users"]]=True
    source=cohort_groups(train,event_indices,known)
    target=cohort_groups(evaluation,np.arange(len(evaluation["users"])),final_known)
    source_counts=np.bincount(source,minlength=10).astype(float)
    target_counts=np.bincount(target,minlength=10).astype(float)
    ratio=(target_counts/target_counts.sum())/np.maximum(source_counts/source_counts.sum(),1/len(source))
    ratio=np.clip(ratio,0.25,4)
    weights=ratio[source];weights/=weights.mean()
    return weights,{"groups":"history length 0,1-4,5-9,10-19,20 crossed with known user",
                    "source_counts":source_counts.tolist(),"evaluation_counts":target_counts.tolist(),
                    "weight_min":float(weights.min()),"weight_max":float(weights.max()),
                    "assumption":"label conditional behavior is sufficiently comparable within these coarse groups; diagnostic only"}


def weighted_auc(labels,predictions,weights):
    return float(roc_auc_score(np.asarray(labels).ravel(),np.asarray(predictions).ravel(),sample_weight=np.repeat(weights,10)))
