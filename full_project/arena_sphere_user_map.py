"""Fixed-item unit-sphere user MAP using all past clicked real slates.

Compare its directional correction to V29's accepted joint user/item update;
the current parent already contains that evidence, so it is not added twice.
Global coefficients are reused from the strictly earlier calibrated recipe.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import time
import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from arena_ood import ROOT,guard,write_json,sha
from arena_metric_geometry import cosine,preference_probabilities
from arena_sphere_user_solver import solve_users

OUT=ROOT/'competition/sphere_user_newton'
RECIPE=dict(version=2,solver='damped_riemannian_newton',fit_window='all_past_clicked_slates',priors=[1.,4.,16.],max_steps=32,recovery_steps=128,
            gradient_threshold=1e-4,reference='conditional_full/prior1.0/geometry_e8.npz',
            reference_mass=.25,masses=[.25,.5,1.],items_fixed=True,objective='conditional_choice_chord_prior_on_sphere')


def normalized(x):
    x=np.asarray(x,dtype='float64')
    return x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)


def reference(day):
    path=ROOT/'competition/conditional_full'/day/'prior1.0/geometry_e8.npz'
    assert path.exists(),'The accepted full-prefix reference must exist before this independent experiment.'
    return dict(np.load(path))


def calibration(day):
    from run import timestamp
    path=ROOT/'competition/conditional_full'/day/'calibration.json'
    r=json.loads(path.read_text())
    assert r['first_fit_timestamp']>=timestamp(r['prefix']) and r['last_fit_timestamp']<timestamp(day)
    return r


def fit(day,prior,raw):
    from run import timestamp
    from arena_core import snapshot
    from arena_latent_delta_export import component
    from arena_conditional_sphere import covariates
    from numba import set_num_threads
    folder=OUT/day/('prior'+str(prior));folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        r=json.loads((folder/'result.json').read_text());assert r['recipe']==RECIPE and r['prior']==prior
        return folder
    set_num_threads(4);rep=component(day,'unitfused25_delta')[1]
    initial,vectors=normalized(rep['users']),normalized(rep['items']);cal=calibration(day)
    coef=np.asarray(cal['coefficients'],dtype='float64')
    fit=np.flatnonzero(raw['timestamps']<timestamp(day))
    fit=fit[raw['labels'][fit].sum(1)>0]
    assert len(fit) and int(raw['timestamps'][fit].max())<timestamp(day)
    fit=fit[np.argsort(raw['users'][fit],kind='stable')]
    users=raw['users'][fit];items=np.asarray(raw['items'][fit]);target=np.asarray(raw['labels'][fit]).argmax(1)
    counts=np.bincount(users,minlength=len(initial));ptr=np.r_[0,np.cumsum(counts)].astype('int64')
    x=sparse.load_npz(snapshot(day)/'interactions.npz')
    support=np.diff(x.indptr).astype('float64');precision=prior*np.maximum(support,1.)
    ib=np.load(snapshot(day)/'stats.npz')['ib'];values=covariates(raw,fit,rep,ib)
    offset=(values[:,:,1:]*coef[1:]).sum(2);del values
    base=offset+coef[0]*np.einsum('bd,bid->bi',initial[users],vectors[items])
    base_loss=float((logsumexp(base,axis=1)-base[np.arange(len(fit)),target]).sum());del base
    started=time.monotonic();u,loss,gradient,steps=solve_users(ptr,items,offset,target,vectors,initial,coef[0],precision,RECIPE['max_steps'])
    recovered=False
    if np.max(gradient)>RECIPE['gradient_threshold']:
        u,loss,gradient,steps=solve_users(ptr,items,offset,target,vectors,initial,coef[0],precision,RECIPE['recovery_steps']);recovered=True
    write_json(folder/'numerical_audit.json',dict(finite=bool(np.isfinite(u).all() and np.isfinite(loss).all()),
               gradient_quantiles=np.quantile(gradient,[.5,.9,.99,1]).tolist(),worst_user=int(np.argmax(gradient)),
               worst_user_events=int(counts[np.argmax(gradient)]),recovery_used=recovered,
               step_quantiles=np.quantile(steps,[.5,.9,.99,1]).tolist(),objective_before=base_loss,objective_after=float(loss.sum())))
    assert np.isfinite(u).all() and np.isfinite(loss).all() and np.max(gradient)<RECIPE['gradient_threshold']
    assert float(loss.sum())<=base_loss+1e-5
    known=np.linalg.norm(initial,axis=1)>0
    np.testing.assert_allclose(np.linalg.norm(u[known],axis=1),1,atol=1e-10,rtol=1e-10)
    np.testing.assert_array_equal(u[counts==0],initial[counts==0])
    np.savez(folder/'geometry.npz',users=u.astype('f'),items=vectors.astype('f'))
    np.savez_compressed(folder/'solver.npz',loss=loss,gradient=gradient,steps=steps,precision=precision)
    result=dict(recipe=RECIPE,day=day,prior=prior,fit_events=len(fit),fitted_users=int((counts>0).sum()),
                max_fit_timestamp=int(raw['timestamps'][fit].max()),fit_indices_sha256=__import__('hashlib').sha256(fit.tobytes()).hexdigest(),
                source_calibration=cal,source_calibration_sha256=sha(ROOT/'competition/conditional_full'/day/'calibration.json'),
                original_objective=base_loss,new_objective=float(loss.sum()),gradient_max=float(np.max(gradient)),
                recovery_used=recovered,step_quantiles=np.quantile(steps,[.5,.9,.99,1]).tolist(),seconds=time.monotonic()-started,
                policy='Fixed source-only item directions. All user fitting targets and teacher calibration strictly before cutoff; no query labels. Exact objective decreases under normalized line-search updates.')
    write_json(folder/'result.json',result);print(json.dumps(dict(stage='user_sphere_fit',**result)),flush=True)
    return folder


def delta(day,prior,raw,indices):
    path=fit(day,prior,raw);new=dict(np.load(path/'geometry.npz'));old=reference(day)
    temperature=calibration(day)['coefficients'][0]
    return RECIPE['reference_mass']*temperature*(cosine(new,raw,indices)-cosine(old,raw,indices))


def campaign():
    from ctr_data import load_cache
    from arena_conditional_sphere import metrics
    from arena_validation import importance_weights
    OUT.mkdir(parents=True,exist_ok=True);before=guard()
    pointer=json.loads((ROOT/'competition/CURRENT_BEST.json').read_text())
    if (OUT/'incumbent_pointer.json').exists():assert json.loads((OUT/'incumbent_pointer.json').read_text())==pointer
    else:write_json(OUT/'incumbent_pointer.json',pointer)
    if (OUT/'recipe.json').exists():assert json.loads((OUT/'recipe.json').read_text())==RECIPE
    else:write_json(OUT/'recipe.json',RECIPE)
    assert not (OUT/'selection.json').exists(),'Use another campaign for subsequent hypotheses.'
    _,raw=load_cache(ROOT/'data/cache_v1','train');results={}
    for day in ['26','25']:
        write_json(OUT/'state.json',dict(stage='fitting',day=day,updated=time.time()))
        parent=np.load((ROOT/pointer['csv']).parent/('validation_'+day+'.npz'));ix=parent['event_indices']
        iw=importance_weights('2025-01-'+day,ix)[0];baseline=metrics(parent['labels'],parent['predictions'],parent['predictions'],iw);rows=[]
        for prior in RECIPE['priors']:
            value=delta('2025-01-'+day,prior,raw,ix)
            np.savez_compressed(OUT/('correction_'+day+'_prior'+str(prior)+'.npz'),event_indices=ix,labels=parent['labels'],correction=value)
            for mass in RECIPE['masses']:
                p=preference_probabilities(parent['predictions'],value,mass)
                row=dict(prior=prior,mass=mass,**metrics(parent['labels'],p,parent['predictions'],iw));rows.append(row)
                print(json.dumps(dict(stage='user_sphere_probe',day=day,**row)),flush=True)
        results[day]=dict(baseline=baseline,rows=rows);write_json(OUT/('result_'+day+'.json'),results[day])
    eligible=[]
    for a,b in zip(results['26']['rows'],results['25']['rows']):
        assert a['prior']==b['prior'] and a['mass']==b['mass']
        ds=[r['auc']-results[d]['baseline']['auc'] for d,r in [('26',a),('25',b)]]
        wd=np.mean([r['weighted_auc']-results[d]['baseline']['weighted_auc'] for d,r in [('26',a),('25',b)]])
        if min(ds)>1e-6 and wd>1e-6:
            eligible.append(dict(prior=a['prior'],mass=a['mass'],auc_26=a['auc'],auc_25=b['auc'],weighted_delta=float(wd),mean_near_top1=.5*(a['near_tie_top1']+b['near_tie_top1'])))
    selected=None
    if eligible:
        best=max(r['weighted_delta'] for r in eligible)
        selected=max([r for r in eligible if r['weighted_delta']>=best-1e-6],key=lambda r:r['mean_near_top1'])
    write_json(OUT/'selection.json',dict(selected=selected,eligible=len(eligible),target_auc=.838))
    write_json(OUT/'state.json',dict(stage='qualified_needs_third_origin' if selected else 'completed_no_promotion',selected=selected,updated=time.time()))
    lines=['# 固定物品表示的球面用户后验','',
           '使用全部过去真实点击候选集合，以固定物品方向、球面用户方向和支持度先验求解条件偏好。数值收敛由拟合目标与梯度决定，不看查询标签。',
           '修正与V29已经采用的联合用户/物品更新做差，避免直接重复叠加同一曝光证据；三个固定先验、三个固定使用幅度。','',
           '| 先验 | 幅度 | Jan26 | Jan25 |','|---|---|---:|---:|']
    for a,b in zip(results['26']['rows'],results['25']['rows']):lines.append(f'| {a["prior"]} | {a["mass"]} | {a["auc"]:.9f} | {b["auc"]:.9f} |')
    lines+=['',f'双切分合格 {len(eligible)} 组；未合格则保留V31，不生成或替换CSV。目标0.838尚未达到。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));assert guard()==before
    print(json.dumps(dict(stage='user_sphere_completed',selected=selected)),flush=True)


if __name__=='__main__':campaign()
