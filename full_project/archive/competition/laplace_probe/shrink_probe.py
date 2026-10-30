import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from arena_laplace_user import model
from arena_bayes_mf import load_model
from ctr_data import load_cache
from arena_ood import ROOT,write_json
_,raw=load_cache(ROOT/'data/cache_v1','train')
report={}
for day in ['26']:
 folder=model('2025-01-'+day);net=load_model(folder,'cpu');saved=np.load(ROOT/'competition/laplace_probe'/day/'validation.npz');ix=saved['event_indices'];labels=saved['labels'].ravel();u=net.u0.numpy()[raw['users'][ix]];v=net.items.weight.detach().numpy()[raw['items'][ix]];delta=net.users.weight.detach().numpy()[raw['users'][ix]]-u;bias=net.user_delta.weight.detach().numpy()[raw['users'][ix],0];scale=float(net.coefficients.detach()[0]);latent=(delta[:,None]*v).sum(2)*scale
 p=saved['prior'].astype('float64');outside=np.maximum(1-p.sum(1),1e-10);logp=np.log(np.concatenate([p,outside[:,None]],1));recipes=[];cosines={}
 for strength in [0.,.1,.25,.5,1.]:
  z=u+strength*delta;c=(z[:,None]*v).sum(2)/(np.linalg.norm(z,axis=1).clip(1e-6)[:,None]*np.linalg.norm(v,axis=2).clip(1e-6));cosines[str(strength)]=float(roc_auc_score(labels,c.ravel()))
  for intercept in [0.,.1,.25,.5,1.]:
   shift=np.concatenate([strength*latent+intercept*bias[:,None],np.zeros((len(ix),1))],1);prob=softmax(logp+shift,axis=1)[:,:10];recipes.append(dict(latent_mass=strength,bias_mass=intercept,auc=float(roc_auc_score(labels,prob.ravel()))))
 report[day]=dict(cosines=cosines,recipes=recipes,best=max(recipes,key=lambda r:r['auc']))
write_json(ROOT/'competition/laplace_probe/shrink_diagnostic.json',report)
print(json.dumps({d:dict(cosines=r['cosines'],best=r['best']) for d,r in report.items()},indent=2),flush=True)
