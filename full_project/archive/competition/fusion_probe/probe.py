import json,time
import numpy as np
from pathlib import Path
from scipy.linalg import qr,svd
from ctr_data import load_cache,temporal_split
from arena_sgns_features import embedding_model
from arena_ood import source
from run import timestamp
from sklearn.metrics import roc_auc_score
out=Path('competition/fusion_probe')
_,raw=load_cache('data/cache_v1','train')
report={}
for day in ['26','25']:
 cutoff='2025-01-'+day;fc=source(cutoff)['feature_config']
 a=embedding_model(cutoff,8,recipe=fc['sgns_recipes']['8'])
 b=embedding_model(cutoff,8,recipe=fc['sgns_extra'][0]['recipe'])
 _,ix=temporal_split(raw,timestamp(cutoff))
 y=raw['labels'][ix].ravel()
 def norm(x):return x/np.linalg.norm(x,axis=-1,keepdims=True).clip(1e-6)
 def scores(u,v):
  z=(u[raw['users'][ix],None]*v[raw['items'][ix]]).sum(2)
  c=(norm(u)[raw['users'][ix],None]*norm(v)[raw['items'][ix]]).sum(2)
  return dict(dot_auc=float(roc_auc_score(y,z.ravel())),cos_auc=float(roc_auc_score(y,c.ravel())))
 r={str(j):scores(m['users'],m['items']) for j,m in enumerate([a,b])}
 for kind in ['raw','cosine']:
  us=[norm(m['users']) if kind=='cosine' else m['users'] for m in [a,b]]
  vs=[norm(m['items']) if kind=='cosine' else m['items'] for m in [a,b]]
  u=np.concatenate(us,1)/np.sqrt(2);v=np.concatenate(vs,1)/np.sqrt(2)
  qu,ru=qr(u,mode='economic');qv,rv=qr(v,mode='economic')
  left,s,right=svd(ru@rv.T,full_matrices=False)
  r[kind]={}
  for rank in [6,8,10,12,16]:
   uu=(qu@left[:,:rank])*np.sqrt(s[:rank]);vv=(qv@right[:rank].T)*np.sqrt(s[:rank])
   uu[0]=vv[0]=0
   r[kind][str(rank)]=scores(uu,vv)
 report[day]=r
 (out/'report.json').write_text(json.dumps(report,indent=2))
 print(day,json.dumps(r),flush=True)
