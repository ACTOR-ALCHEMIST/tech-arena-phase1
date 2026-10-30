"""Past-only user preference MAP with fixed spherical item factors.

Global coefficients are calibrated in an earlier frame. The final per-user
linear slate likelihood is strictly convex under a positive Gaussian prior.
Curvature is an empirical-prior approximation, not calibrated confidence.
"""
import fcntl
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy import sparse
from scipy.special import expit, logsumexp
from arena_ood import ROOT, sha, write_json
from arena_bayes_mf import BayesianMF, load_model, predict
from arena_laplace_solver import solve_users

RECIPE = dict(version=1, calibration_epochs=4, calibration_seed=7153, calibration_lr=.003,
              initial_dot=2., prior_strength=1., latent_prior_intercept=1., latent_prior_per_click=.5,
              bias_prior_intercept=2., bias_prior_per_event=.63*.37, newton_steps=12, dimension=8)
TAG = 'laplace_user_v1_s1'


def initial(cutoff):
    from arena_core import snapshot
    from arena_spherical_repr import model
    rep = dict(np.load(model(cutoff)/'factors_e32.npz'))
    x = sparse.load_npz(snapshot(cutoff)/'interactions.npz')
    stats = dict(np.load(snapshot(cutoff)/'stats.npz'))
    uc, ic = np.diff(x.indptr), np.asarray(x.sum(0)).ravel()
    ancillary = dict(ub=stats['ub'], ib=stats['ib'], u_precision=1+.5*uc,
                     v_precision=1+.5*ic, bu_precision=2+(stats['ui']/10)*.63*.37,
                     bi_precision=4+stats['ii']*float(stats['ctr'])*(1-float(stats['ctr'])),
                     head_users=stats['ui']/10, head_items=stats['ii'])
    net = BayesianMF(rep, ancillary, dict(fine_items=False, anchor_scale=1., representation='spherical', posterior='laplace_user'))
    with torch.no_grad():
        net.coefficients[0] = 2.
    return net, uc, stats


@torch.inference_mode()
def score_matrix(net, raw, indices, feature_values=False):
    from run import get_batch
    shape = (len(indices), 10, 16) if feature_values else (len(indices), 11)
    out = np.empty(shape, 'float32')
    net.cpu().eval()
    for start in range(0, len(indices), 2048):
        stop = min(start+2048, len(indices))
        batch = get_batch(raw, indices[start:stop], 'cpu', labels=False)
        value = net.features(batch) if feature_values else net(batch)
        out[start:stop] = value.numpy()
    return out


def model(cutoff):
    from arena_core import snapshot
    from arena_supervised_repr import training_windows
    from ctr_data import load_cache
    from run import timestamp
    folder = snapshot(cutoff)/TAG
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            assert json.loads((folder/'result.json').read_text())['recipe'] == RECIPE
            return folder
        tick = time.monotonic()
        meta, raw = load_cache(ROOT/'data/cache_v1', 'train')
        prefix, fit, calibration = training_windows(raw, cutoff)
        assert len(fit) and len(calibration) and int(raw['timestamps'][fit].max()) < timestamp(cutoff)
        torch.manual_seed(7153)
        torch.set_num_threads(2)
        net, _, _ = initial(prefix)
        net.configure(False)
        values = torch.from_numpy(score_matrix(net, raw, calibration, feature_values=True))
        labels = np.asarray(raw['labels'][calibration])
        target = torch.from_numpy(np.where(labels.sum(1)>0, labels.argmax(1), 10)).long()
        opt = torch.optim.AdamW([net.coefficients, net.outside], lr=.003, weight_decay=.01)
        rng, records = np.random.default_rng(7153), []
        for epoch in range(1, 5):
            total, start_time = 0., time.monotonic()
            order = rng.permutation(len(calibration))
            for start in range(0, len(order), 2048):
                chosen = torch.from_numpy(order[start:start+2048])
                score = (values[chosen]*net.coefficients).sum(2)
                z = torch.cat([score, net.outside.expand(len(chosen), 1)], 1)
                loss = F.cross_entropy(z, target[chosen])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([net.coefficients, net.outside], 5)
                opt.step()
                total += loss.item()*len(chosen)
            row = dict(phase='calibration', epoch=epoch, loss=total/len(calibration), seconds=time.monotonic()-start_time)
            records.append(row)
            print(json.dumps(dict(stage='laplace_user', cutoff=cutoff, **row)), flush=True)
        coefficients, outside = net.coefficients.detach().clone(), net.outside.detach().clone()
        del net, values, target, opt
        gc.collect()
        net, support, stats = initial(cutoff)
        with torch.no_grad():
            net.coefficients.copy_(coefficients)
            net.outside.copy_(outside)
        net.requires_grad_(False)
        order = np.argsort(raw['users'][fit], kind='stable')
        fit = fit[order]
        users = np.asarray(raw['users'][fit])
        items = np.asarray(raw['items'][fit])
        labels = np.asarray(raw['labels'][fit])
        assert (labels.sum(1)<=1).all()
        target = np.where(labels.sum(1)>0, labels.argmax(1), 10).astype('int64')
        offsets = score_matrix(net, raw, fit).astype('float64')
        n_users = len(net.u0)
        ptr = np.r_[0, np.cumsum(np.bincount(users, minlength=n_users))].astype('int64')
        precision = np.empty((n_users, 9), 'float64')
        precision[:, :8] = (1+.5*support)[:, None]
        precision[:, 8] = 2+(stats['ui']/10)*.63*.37
        precision *= RECIPE['prior_strength']
        base_loss = float((logsumexp(offsets, axis=1)-offsets[np.arange(len(fit)), target]).sum())
        vectors = net.items.weight.detach().numpy().astype('float64')
        scale = float(net.coefficients[0])
        start_time = time.monotonic()
        delta, covariance, loss, gradient, steps = solve_users(ptr, items, offsets, target, vectors, scale, precision, 12)
        assert np.isfinite(delta).all() and np.isfinite(covariance).all()
        assert loss.sum() <= base_loss+1e-5
        assert np.quantile(gradient, .99) < 1e-4, 'Newton solver did not sufficiently converge'
        covariance = .5*(covariance+covariance.transpose(0, 2, 1))
        assert np.linalg.eigvalsh(covariance.astype('float64')).min() > 0
        with torch.no_grad():
            net.users.weight.add_(torch.from_numpy(delta[:, :8]))
            net.user_delta.weight[:, 0].copy_(torch.from_numpy(delta[:, 8]))
        torch.save(dict(state_dict=net.state_dict(), config=net.config, cache_id=meta['cache_id']), folder/'model.tmp.pt')
        os.replace(folder/'model.tmp.pt', folder/'model.pt')
        np.savez_compressed(folder/'covariance.tmp.npz', covariance=covariance, precision=precision.astype('float32'), delta=delta)
        os.replace(folder/'covariance.tmp.npz', folder/'covariance.npz')
        write_json(folder/'result.json', dict(recipe=RECIPE, cutoff=cutoff, calibration_prefix=prefix,
                   cache_id=meta['cache_id'], fitting_events=len(fit), calibration_events=len(calibration),
                   max_fitting_timestamp=int(raw['timestamps'][fit].max()), max_calibration_timestamp=int(raw['timestamps'][calibration].max()),
                   fitting_indices_sha256=hashlib.sha256(fit.tobytes()).hexdigest(), calibration=records,
                   baseline_loss=base_loss, posterior_objective=float(loss.sum()),
                   newton_seconds=time.monotonic()-start_time, gradient_quantiles=np.quantile(gradient, [.5,.9,.99,1.]).tolist(),
                   step_quantiles=np.quantile(steps,[.5,.9,1.]).tolist(), coefficients=coefficients.tolist(), outside=float(outside),
                   model_sha256=sha(folder/'model.pt'), covariance_sha256=sha(folder/'covariance.npz'), seconds=time.monotonic()-tick,
                   policy='All accessed exposure labels strictly before cutoff; earlier calibration frame. Empirical prior reuses fitting click information; local covariance is not calibrated confidence.'))
        print(json.dumps(dict(stage='laplace_user_solved', cutoff=cutoff, objective_drop=base_loss-float(loss.sum()),
                             gradient_max=float(gradient.max()), seconds=time.monotonic()-tick)), flush=True)
    return folder


def predictive_uncertainty(probabilities, design, covariance):
    """Rotation-invariant second-order Gaussian approximation to softmax."""
    p = np.asarray(probabilities, 'float64')
    outside = np.maximum(1-p.sum(1), 1e-10)
    full = np.concatenate([p, outside[:, None]], 1)
    full /= full.sum(1, keepdims=True)
    p = full[:, :-1]
    mean = np.einsum('bi,bid->bd', p, design)
    centered = design-mean[:, None]
    linear = np.einsum('bid,bdk,bik->bi', design, covariance, design).clip(0)
    center = np.einsum('bid,bdk,bik->bi', centered, covariance, centered).clip(0)
    outside_var = np.einsum('bd,bdk,bk->b', mean, covariance, mean).clip(0)
    average = (p*center).sum(1)+full[:, -1]*outside_var
    full_var = np.concatenate([center, outside_var[:, None]], 1)
    corrected = (full*(1+.5*(full_var-average[:, None]))).clip(1e-10)
    corrected /= corrected.sum(1, keepdims=True)
    return linear, center, corrected[:, :-1], p*np.sqrt(center)


def features_for_posterior(net, covariance, data, indices):
    from arena_supervised_repr import features_for_model
    x, names = features_for_model(net, data, indices, 'laplace')
    users, items = np.asarray(data['users'][indices]), np.asarray(data['items'][indices])
    vectors = net.items.weight.detach().numpy()[items]
    scale = float(net.coefficients.detach()[0])
    design = np.concatenate([vectors*scale, np.ones((*items.shape, 1), 'float32')], 2).astype('float64')
    cov = np.asarray(covariance[users], 'float64')
    p = expit(x[:, -1].astype('float64')).reshape(items.shape)
    linear, center, pp, std = predictive_uncertainty(p, design, cov)
    delta = net.users.weight.detach().numpy()-net.u0.numpy()
    bias = net.user_delta.weight.detach().numpy()[:, 0]
    change = np.einsum('bd,bid->bi', delta[users], vectors)*scale+bias[users, None]
    logit = lambda p:np.log(p.clip(1e-7,1-1e-7))-np.log1p(-p.clip(1e-7,1-1e-7))
    expand = lambda a:np.broadcast_to(a[:, None], items.shape)
    extra = [expand(np.linalg.norm(delta[users],axis=1)), expand(bias[users]),
             expand(np.trace(cov[:, :-1, :-1],axis1=1,axis2=2)), expand(np.linalg.slogdet(cov)[1]),
             np.log1p(linear), center, linear-linear.mean(1,keepdims=True), change, change-change.mean(1,keepdims=True),
             logit(pp), logit(pp)-logit(p), std]
    result = np.concatenate([x,np.stack(extra,2).reshape(-1,12)],1).astype('float32')
    names += ['laplace_'+n for n in ['delta_norm','delta_bias','cov_trace','cov_logdet','log_score_variance','logp_variance',
              'variance_relative','score_change','score_change_relative','predictive_logit','predictive_logit_delta','probability_std']]
    assert result.shape==(len(indices)*10,36) and np.isfinite(result).all()
    return result,names


def extra_features(data, indices, cutoff, config=None):
    folder = model(cutoff)
    net = load_model(folder, 'cpu')
    cov = np.load(folder/'covariance.npz')['covariance']
    pieces, names = [], []
    for start in range(0,len(indices),2048):
        x,names=features_for_posterior(net,cov,data,indices[start:start+2048])
        pieces.append(x)
    return np.concatenate(pieces,0),names


if __name__=='__main__':
    import argparse
    from ctr_data import load_cache,temporal_split
    from arena_core import evaluate
    p=argparse.ArgumentParser()
    p.add_argument('--cutoff',default='2025-01-26')
    args=p.parse_args()
    folder=model(args.cutoff)
    _,raw=load_cache(ROOT/'data/cache_v1','train')
    _,ix=temporal_split(raw,__import__('run').timestamp(args.cutoff))
    net=load_model(folder,'cpu')
    updated=predict(net,raw,ix,'cpu')
    baseline=net.users.weight.detach().clone()
    bias=net.user_delta.weight.detach().clone()
    with torch.no_grad():
        net.users.weight.copy_(net.u0)
        net.user_delta.weight.zero_()
    prior=predict(net,raw,ix,'cpu')
    with torch.no_grad():
        net.users.weight.copy_(baseline)
        net.user_delta.weight.copy_(bias)
    x,names=extra_features(raw,ix,args.cutoff)
    out=ROOT/'competition/laplace_probe'/args.cutoff[-2:]
    out.mkdir(parents=True,exist_ok=True)
    record=dict(cutoff=args.cutoff,prior=evaluate(raw['labels'][ix],prior),posterior=evaluate(raw['labels'][ix],updated),
                predictive=evaluate(raw['labels'][ix],expit(x[:,-3]).reshape(-1,10)),features=x.shape[1],fitting_metadata=json.loads((folder/'result.json').read_text()))
    write_json(out/'report.json',record)
    np.savez_compressed(out/'validation.npz',event_indices=ix,labels=raw['labels'][ix],prior=prior,predictions=updated,predictive=expit(x[:,-3]).reshape(-1,10))
    print(json.dumps({k:v for k,v in record.items() if k!='fitting_metadata'}),flush=True)
