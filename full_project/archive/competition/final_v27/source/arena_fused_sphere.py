"""Past-only orthogonal fusion of long spherical and independent SGNS views."""
import fcntl
import json
import os
import numpy as np
from scipy import sparse
from arena_ood import write_json,sha
from arena_core import snapshot
from arena_aligned_repr import rotation

RECIPE=dict(version=1,dimension=8,long_epoch=128,long_masses=[.5,.75],
            other_recipe=dict(epochs=450,lr=.01,lr_decay=False,seed=42,
                              reuse_experiment='sgns8_const_n8_binary_e640_s42'),
            alignment='unit_item_minimum_popularity_capped20',retain_agreement_norms=True)
TAG='fused_sphere_v1_long128_s42'


def fuse(a,b,known_users,pop,mass):
    norm=lambda x:x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-8)
    au,av=norm(a['users'].copy()),norm(a['items'].copy())
    bu,bv=norm(b['users'].copy()),norm(b['items'].copy())
    bu[~known_users]=0;bv[pop<=0]=0
    au[~known_users]=0;av[pop<=0]=0
    q=rotation(bv,av,np.minimum(pop,20))
    # Do not renormalize the mixture here: its norm measures cross-view
    # agreement. The standard descriptor separately normalizes cosine terms.
    users=(mass*au+(1-mass)*(bu@q)).astype('f')
    items=(mass*av+(1-mass)*(bv@q)).astype('f')
    users[0]=items[0]=0
    bias=a['bias'].astype('f').copy();bias[pop<=0]=0;bias[0]=0
    return dict(users=users,items=items,bias=bias,rotation=q)


def model(cutoff):
    from arena_spherical_long import model as long_model
    from arena_sgns_features import embedding_model
    from run import timestamp
    path=snapshot(cutoff);folder=path/TAG;folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            assert json.loads((folder/'result.json').read_text())['recipe']==RECIPE
            return folder
        a_path=long_model(cutoff)/'factors_e128.npz';a=dict(np.load(a_path))
        b=embedding_model(cutoff,8,recipe=RECIPE['other_recipe'])
        x=sparse.load_npz(path/'interactions.npz');known=np.diff(x.indptr)>0;pop=np.asarray(x.sum(0)).ravel()
        known[0]=False;pop[0]=0
        for mass in RECIPE['long_masses']:
            rep=fuse(a,b,known,pop,mass)
            np.savez(folder/('w'+str(round(100*mass))+'.tmp.npz'),**rep)
            os.replace(folder/('w'+str(round(100*mass))+'.tmp.npz'),folder/('w'+str(round(100*mass))+'.npz'))
        metadata=json.loads((path/'metadata.json').read_text());assert metadata['max_fit_timestamp']<timestamp(cutoff)
        write_json(folder/'result.json',dict(cutoff=cutoff,recipe=RECIPE,max_fit_timestamp=metadata['max_fit_timestamp'],
                   long_source_sha256=sha(a_path),policy='Only past embedding tables and past popularity used for orthogonal fit and fusion. No query labels accessed.'))
    return folder


def extra_features(data,indices,cutoff,config=None):
    from arena_coclick import describe
    from run import timestamp
    assert np.all(data['timestamps'][indices]>=timestamp(cutoff))
    folder=model(cutoff);parts=[];names=[]
    for mass in RECIPE['long_masses']:
        tag='w'+str(round(100*mass));rep=dict(np.load(folder/(tag+'.npz')))
        x,cols=describe(rep,data,indices);parts.append(x)
        names += [c.replace('coclick_','fusedsphere_'+tag+'_',1) for c in cols]
    return np.concatenate(parts,1),names
