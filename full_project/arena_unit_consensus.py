"""Nonlinear ranking features from the accepted two-unit-sphere consensus.

This is distinct from V25's raw-SGNS plus sphere mixture. The strong unit/long
mixture here was independently accepted as a direct correction in V26.
"""
import numpy as np
from arena_coclick import describe
from arena_metric_geometry import cosine


def from_representations(raw,ix,old,new,long):
    a,names=describe(new,raw,ix);b,old_names=describe(old,raw,ix)
    assert names==old_names
    cn=cosine(new,raw,ix);co=cosine(old,raw,ix);cl=cosine(long,raw,ix)
    d=cn-co;e=cn-cl
    extra=[d,d-d.mean(1,keepdims=True),e,e-e.mean(1,keepdims=True)]
    extra_names=['cos_delta_old','cos_delta_old_relative','cos_delta_long','cos_delta_long_relative']
    for name in ['profile_cos','hist_mean','last3','decay24','hist_max']:
        k=names.index('coclick_'+name);extra.append((a[:,k]-b[:,k]).reshape(len(ix),10))
        extra_names.append(name+'_delta_old')
    x=np.concatenate([a,np.stack(extra,2).reshape(-1,len(extra))],1).astype('f')
    cols=[n.replace('coclick_','unit_consensus_',1) for n in names]+['unit_consensus_'+n for n in extra_names]
    assert x.shape==(len(ix)*10,31) and np.isfinite(x).all()
    return x,cols


def extra_features(raw,ix,cutoff,config=None):
    from arena_latent_delta_export import component
    from arena_spherical_long import model
    from run import timestamp
    assert np.all(raw['timestamps'][ix]>=timestamp(cutoff))
    old,new=component(cutoff,'unitfused25_delta');long=dict(np.load(model(cutoff)/'factors_e128.npz'))
    return from_representations(raw,ix,old,new,long)
