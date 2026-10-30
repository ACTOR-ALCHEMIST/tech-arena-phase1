"""Fresh, past-only SGNS refinement under real exposure negatives.

Calibrate the score scale in an earlier frame, freeze it, and refine a fresh
snapshot under ID-support-weighted priors. The representation builder never
reads validation labels. Epoch4/16 were fixed by separate Jan26/25 probes.
"""
import copy
import fcntl
import hashlib
import time
from datetime import datetime, timedelta
import numpy as np
import torch
from torch.nn import functional as F
from arena_ood import ROOT, sha, write_json
from arena_bayes_mf import initialize, load_model, predict

RECIPE = dict(version=1, dimension=8, calibration_epochs=4, calibration_seed=7153,
              calibration_lr=.003, refinement_epochs=16, refinement_seed=1727,
              lr=.001, anchor_scale=1., ema=.99, outputs=[4, 16])


def calibration_prefix(cutoff):
    prefix = (datetime.strptime(cutoff, '%Y-%m-%d')-timedelta(days=3)).strftime('%Y-%m-%d')
    prefix = max('2025-01-21', prefix)
    if prefix >= cutoff:
        raise ValueError('No honest earlier calibration window for '+cutoff)
    return prefix


def training_windows(raw, cutoff):
    from run import timestamp
    prefix = calibration_prefix(cutoff)
    fit = np.flatnonzero(raw['timestamps'] < timestamp(cutoff))
    calibration = fit[raw['timestamps'][fit] >= timestamp(prefix)]
    assert len(fit) and len(calibration)
    assert int(raw['timestamps'][fit].max()) < timestamp(cutoff)
    return prefix, fit, calibration


def model(cutoff):
    from arena_core import snapshot
    from ctr_data import load_cache
    from run import get_batch
    import json
    folder = snapshot(cutoff)/'supervised_fresh_v1_d8_e16'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        meta, raw = load_cache(ROOT/'data/cache_v1', 'train')
        if (folder/'result.json').exists() and all((folder/('e'+str(e))/'model.pt').exists() for e in RECIPE['outputs']):
            saved = json.loads((folder/'result.json').read_text())
            assert saved['recipe'] == RECIPE and saved['cache_id'] == meta['cache_id']
            return folder
        prefix, fit, calibration = training_windows(raw, cutoff)
        config = dict(anchor_scale=1., fine_items=True)
        torch.set_num_threads(2)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        torch.manual_seed(RECIPE['calibration_seed'])
        calibration_model = initialize(raw, calibration, cutoff, config, prefix=prefix).to(device)
        calibration_model.configure(False)
        opt = torch.optim.AdamW([calibration_model.coefficients, calibration_model.outside], lr=.003, weight_decay=.01)
        rng, records = np.random.default_rng(RECIPE['calibration_seed']), []
        started = time.monotonic()
        for epoch in range(1, 5):
            tick, total = time.monotonic(), 0.
            order = rng.permutation(calibration)
            for start in range(0, len(order), 2048):
                indices = order[start:start+2048]
                batch = get_batch(raw, indices, device, labels=True)
                y = batch['labels']
                loss = F.cross_entropy(calibration_model(batch), torch.where(y.sum(1)>0, y.argmax(1), 10))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(calibration_model.parameters(), 5)
                opt.step()
                total += loss.item()*len(indices)
            row = dict(phase='calibration', epoch=epoch, loss=total/len(calibration), seconds=time.monotonic()-tick)
            records.append(row)
            print(__import__('json').dumps(dict(stage='supervised_fresh', cutoff=cutoff, **row)), flush=True)
        coefficients, outside = calibration_model.coefficients.detach().clone(), calibration_model.outside.detach().clone()
        del calibration_model, opt
        if device.type == 'mps':
            torch.mps.empty_cache()
        torch.manual_seed(RECIPE['refinement_seed'])
        net = initialize(raw, fit, cutoff, config, prefix=cutoff).to(device)
        with torch.no_grad():
            net.coefficients.copy_(coefficients)
            net.outside.copy_(outside)
        net.configure(True)
        ema = copy.deepcopy(net).eval()
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=.001, weight_decay=0)
        rng = np.random.default_rng(RECIPE['refinement_seed'])
        for epoch in range(1, 17):
            tick, total = time.monotonic(), 0.
            order = rng.permutation(fit)
            for start in range(0, len(order), 2048):
                indices = order[start:start+2048]
                batch = get_batch(raw, indices, device, labels=True)
                y = batch['labels']
                ce = F.cross_entropy(net(batch), torch.where(y.sum(1)>0, y.argmax(1), 10))
                loss = ce+net.penalty(batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                opt.step()
                total += loss.item()*len(indices)
                with torch.no_grad():
                    for target, p in zip(ema.parameters(), net.parameters()):
                        if p.requires_grad:
                            target.mul_(.99).add_(p, alpha=.01)
            row = dict(phase='refinement', epoch=epoch, loss=total/len(fit), seconds=time.monotonic()-tick)
            records.append(row)
            if epoch in RECIPE['outputs']:
                path = folder/('e'+str(epoch))
                path.mkdir(exist_ok=True)
                torch.save(dict(state_dict=ema.state_dict(), config=config, cache_id=meta['cache_id']), path/'model.pt')
            if epoch % 4 == 0:
                print(__import__('json').dumps(dict(stage='supervised_fresh', cutoff=cutoff, **row)), flush=True)
                write_json(folder/'training.json', records)
        write_json(folder/'result.json', dict(recipe=RECIPE, cache_id=meta['cache_id'], cutoff=cutoff,
                   calibration_prefix=prefix, calibration_events=len(calibration), fitting_events=len(fit),
                   max_calibration_timestamp=int(raw['timestamps'][calibration].max()),
                   max_fitting_timestamp=int(raw['timestamps'][fit].max()),
                   fitting_indices_sha256=hashlib.sha256(fit.tobytes()).hexdigest(),
                   model_sha256={str(e):sha(folder/('e'+str(e))/'model.pt') for e in RECIPE['outputs']},
                   seconds=time.monotonic()-started, label_policy='All accessed fitting labels strictly before cutoff; no validation labels are read.'))
    return folder


def features_for_model(model, data, indices, tag):
    """Query features use IDs/history/context only; no target column required."""
    from arena_coclick import describe
    model = model.cpu().eval()
    c = model.coefficients.detach().numpy()
    rep = dict(users=model.users.weight.detach().numpy(), items=model.items.weight.detach().numpy(),
               bias=model.ib.numpy()*c[2]+model.item_delta.weight.detach().numpy().ravel())
    x, names = describe(rep, data, indices)
    user_offset = model.ub.numpy()*c[1]+model.user_delta.weight.detach().numpy().ravel()
    offset = np.repeat(user_offset[data['users'][indices]], 10)[:, None]
    p = predict(model, data, indices, torch.device('cpu')).ravel().astype(np.float64).clip(1e-7, 1-1e-7)
    log_odds = (np.log(p)-np.log1p(-p))[:, None]
    result = np.concatenate([x, offset, log_odds], 1).astype('f')
    columns = [n.replace('coclick_', tag+'_', 1) for n in names]+[tag+'_user_offset', tag+'_event_logit']
    assert result.shape == (len(indices)*10, 24) and np.isfinite(result).all()
    return result, columns


def extra_features(data, indices, cutoff, config=None):
    folder = model(cutoff)
    torch.set_num_threads(2)
    pieces, names = [], []
    for epoch in RECIPE['outputs']:
        x, cols = features_for_model(load_model(folder/('e'+str(epoch)), 'cpu'), data, indices, 'fresh'+str(epoch))
        pieces.append(x)
        names += cols
    return np.concatenate(pieces, 1), names
