"""Past-only spherical SGNS, with fixed epoch16/32 representation views."""
import copy
import fcntl
import hashlib
import json
import os
import shutil
import time
import numpy as np
import torch
from scipy import sparse
from arena_ood import ROOT, sha, write_json
from arena_spherical_probe import Spherical

RECIPE = dict(version=1, factors=8, epochs=32, outputs=[16, 32], seed=82719,
              lr=.002, minimum_lr=.0002, weight_decay=.001, ema=.99,
              negatives=8, batch_size=8192, anchor=.02, initial_temperature=6., device='cpu')


def model(cutoff):
    from arena_core import snapshot
    from arena_sgns_features import embedding_model
    from run import timestamp
    path = snapshot(cutoff)
    folder = path/'spherical_v1_d8_e32'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder/'result.json').exists():
            assert json.loads((folder/'result.json').read_text())['recipe'] == RECIPE
            return folder
        metadata = json.loads((path/'metadata.json').read_text())
        assert metadata['max_fit_timestamp'] < timestamp(cutoff)
        tick = time.monotonic()
        probe = ROOT/'competition/spherical_probe'/cutoff[-2:]
        reuse = (cutoff in ['2025-01-26', '2025-01-25'] and (probe/'report.json').exists())
        if reuse:
            # These exact snapshots were fixed before diagnostic metrics were
            # computed. Their metrics do not control the checkpoint selection.
            report = json.loads((probe/'report.json').read_text())
            assert report['epochs'] == 32 and report['seed'] == RECIPE['seed']
            assert report['snapshot_metadata'] == metadata
            for epoch in RECIPE['outputs']:
                shutil.copy2(probe/('factors_e'+str(epoch)+'.npz'), folder/('factors_e'+str(epoch)+'.npz'))
            source = dict(probe=str(probe.relative_to(ROOT)), report_sha256=sha(probe/'report.json'))
        else:
            initial = embedding_model(cutoff, 8, recipe=dict(epochs=320, lr=.01, lr_decay=False, seed=2026,
                                                           reuse_experiment='sgns8_const_e320_s2026'))
            x = sparse.load_npz(path/'interactions.npz')
            ku, ki = np.diff(x.indptr)>0, np.asarray(x.sum(0)).ravel()>0
            ku[0] = ki[0] = False
            pu, pi = x.nonzero()
            torch.manual_seed(RECIPE['seed'])
            torch.set_num_threads(2)
            net = Spherical(initial, ku, ki)
            ema = copy.deepcopy(net).eval()
            opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=32, eta_min=.0002)
            pu, pi = torch.tensor(pu, dtype=torch.long), torch.tensor(pi, dtype=torch.long)
            records = []
            for epoch in range(1, 33):
                start_time, total = time.monotonic(), 0.
                order = torch.randperm(len(pu))
                for start in range(0, len(order), 8192):
                    take = order[start:start+8192]
                    uid, iid = pu[take], pi[take]
                    negative = torch.randint(1, len(initial['items']), (len(take), 8))
                    loss = net.loss(uid, iid, negative)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                    opt.step()
                    with torch.no_grad():
                        for target, p in zip(ema.parameters(), net.parameters()):
                            target.mul_(.99).add_(p, alpha=.01)
                    total += loss.item()*len(take)
                scheduler.step()
                row = dict(epoch=epoch, loss=total/len(pu), seconds=time.monotonic()-start_time)
                records.append(row)
                if epoch in RECIPE['outputs']:
                    tmp = folder/('factors_e'+str(epoch)+'.tmp.npz')
                    np.savez(tmp, **ema.representation())
                    os.replace(tmp, folder/('factors_e'+str(epoch)+'.npz'))
                if epoch % 8 == 0:
                    write_json(folder/'training.json', records)
                    print(json.dumps(dict(stage='spherical_representation', cutoff=cutoff, **row)), flush=True)
            source = dict(binary_interactions=int(x.nnz))
        write_json(folder/'result.json', dict(recipe=RECIPE, cutoff=cutoff, cache_id=metadata['cache_id'],
                   max_fitting_timestamp=metadata['max_fit_timestamp'], source=source,
                   fitting_policy='Only the snapshot click matrix is used; snapshot labels and histories strictly before cutoff. Query targets never accessed.',
                   model_sha256={str(e):sha(folder/('factors_e'+str(e)+'.npz')) for e in RECIPE['outputs']},
                   seconds=time.monotonic()-tick))
    return folder


def extra_features(data, indices, cutoff, config=None):
    from arena_coclick import describe
    folder = model(cutoff)
    pieces, names = [], []
    for epoch in RECIPE['outputs']:
        rep = dict(np.load(folder/('factors_e'+str(epoch)+'.npz')))
        x, cols = describe(rep, data, indices)
        pieces.append(x)
        names += [n.replace('coclick_', 'sphere'+str(epoch)+'_', 1) for n in cols]
    return np.concatenate(pieces, 1), names
