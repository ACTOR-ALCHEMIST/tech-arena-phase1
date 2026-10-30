"""Descriptive past-only test for time-dependent latent preferences."""
import json
from pathlib import Path
import numpy as np
from arena_feedback import feedback
from arena_spherical_repr import model
from run import timestamp
cutoff='2025-01-26'
obs=feedback(cutoff)
order=np.lexsort((obs['minutes'],obs['users']))
u,i,t=[obs[k][order] for k in ['users','items','minutes']]
assert int(t.max())*60 < timestamp(cutoff)
rep=dict(np.load(model(cutoff)/'factors_e32.npz'))
counts=np.bincount(u,minlength=len(rep['users']))
xtx=np.zeros((8,8));xty=np.zeros((8,2));means={};pairs=0
bounds=np.array([0,6,12,24,48,96,168,np.inf])
for gap in range(1,9):
 valid=(u[:-gap]==u[gap:])&(i[:-gap]!=i[gap:])&(t[:-gap]<t[gap:])
 positions=np.flatnonzero(valid)
 for start in range(0,len(positions),32768):
  a=positions[start:start+32768];b=a+gap
  age=(t[b]-t[a])/60
  same_bucket=((t[a]%1440)//360)==((t[b]%1440)//360)
  da=(((t[a]//1440+3)%7)>=5);db=(((t[b]//1440+3)%7)>=5)
  same_day=da==db
  va,vb=rep['items'][i[a]],rep['items'][i[b]]
  profile=rep['users'][u[a]]
  dot=(va*vb).sum(1)
  residual=dot-(va*profile).sum(1)*(vb*profile).sum(1)
  x=np.stack([np.ones(len(a)),np.log1p(age),np.exp(-age/6),np.exp(-age/24),np.exp(-age/168),np.cos(2*np.pi*age/24),same_bucket,same_day],1)
  weight=1/np.maximum(counts[u[a]],1)
  xtx+=x.T@(x*weight[:,None]);xty+=x.T@(np.stack([dot,residual],1)*weight[:,None]);pairs+=len(a)
  bins=np.searchsorted(bounds,age,side='right')-1
  for k in range(len(bounds)-1):
   for same in [False,True]:
    keep=(bins==k)&(same_bucket==same)
    key=f'{bounds[k]}-{bounds[k+1]}h_same{int(same)}'
    total=means.setdefault(key,dict(pairs=0,weight=0.,dot_sum=0.,residual_sum=0.))
    total['pairs']+=int(keep.sum());total['weight']+=float(weight[keep].sum());total['dot_sum']+=float((dot[keep]*weight[keep]).sum());total['residual_sum']+=float((residual[keep]*weight[keep]).sum())
coef=np.linalg.solve(xtx,xty)
for row in means.values():
 row['mean_cosine']=row['dot_sum']/max(row['weight'],1e-12);row['mean_residual_dot']=row['residual_sum']/max(row['weight'],1e-12)
report=dict(cutoff=cutoff,max_click_timestamp=int(t.max())*60,unique_past_clicks=len(u),nonrepeat_pairs=pairs,features=['intercept','log_age','decay6','decay24','decay168','cos_24h','same_bucket4','same_day_type'],weighted_OLS={'cosine':coef[:,0].tolist(),'residual_dot':coef[:,1].tolist(),'condition_number':float(np.linalg.cond(xtx))},age_bin_means=means,limitation='Descriptive fitting-history association only, not a causal effect or held-out model gain; each pair weighted inverse user click count; repeated same-item pairs removed.')
Path('competition/temporal_signal_probe/report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ['unique_past_clicks','nonrepeat_pairs','weighted_OLS']},indent=2),flush=True)
