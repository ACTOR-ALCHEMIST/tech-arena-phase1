"""Past-only full-prefix negative refinement of already-fresh SGNS factors.

Read-only diagnostic: fixed calibration learned exclusively before the cutoff;
no CSV, selection, or incumbent changes. Epochs are reported, not auto-promoted.
"""
import copy
import argparse
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from arena_bayes_mf import initialize, predict
from arena_ood import ROOT, sha, write_json


def run(cutoff='2025-01-26'):
    from ctr_data import load_cache, temporal_split
    from run import timestamp, get_batch
    out = ROOT/'competition'/('fresh_mf_probe' if cutoff == '2025-01-26' else 'fresh_mf_probe_'+cutoff[-2:])
    out.mkdir(parents=True, exist_ok=True)
    _, raw = load_cache(ROOT/'data/cache_v1', 'train')
    fit, valid = temporal_split(raw, timestamp(cutoff))
    source = ROOT/'competition/ood/runs/v12_map_loose'/cutoff/'model.pt'
    s = torch.load(source, map_location='cpu', weights_only=True)
    config = dict(anchor_scale=1., fine_items=True)
    torch.manual_seed(1727)
    torch.set_num_threads(2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = initialize(raw, fit, cutoff, config, prefix=cutoff).to(device)
    with torch.no_grad():
        model.coefficients.copy_(s['state_dict']['coefficients'].to(device))
        model.outside.copy_(s['state_dict']['outside'].to(device))
    model.configure(True)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
    rng, records = np.random.default_rng(1727), []
    report = dict(cutoff=cutoff, fitting_events=len(fit), max_fitting_timestamp=int(raw['timestamps'][fit].max()),
                  calibration_source=str(source.relative_to(ROOT)), calibration_source_sha256=sha(source),
                  protocol='Fresh factors at the stated cutoff; calibration coefficients frozen after fitting earlier honest exposures; all refinements use labels strictly before the cutoff. Diagnostic only.', epochs=records)
    uid, ids, labels = raw['users'][valid], raw['items'][valid], raw['labels'][valid]
    started = time.monotonic()
    for epoch in range(17):
        if epoch:
            model.train()
            order, total = rng.permutation(fit), 0.
            for start in range(0, len(order), 2048):
                selected = order[start:start+2048]
                batch = get_batch(raw, selected, device, labels=True)
                y = batch['labels']
                ce = F.cross_entropy(model(batch), torch.where(y.sum(1)>0, y.argmax(1), 10))
                loss = ce+model.penalty(batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                total += loss.item()*len(selected)
                with torch.no_grad():
                    for target, p in zip(ema.parameters(), model.parameters()):
                        if p.requires_grad:
                            target.mul_(.99).add_(p, alpha=.01)
        if epoch in [0, 2, 4, 8, 16]:
            u, v = ema.users.weight.detach().cpu().numpy(), ema.items.weight.detach().cpu().numpy()
            dot = np.einsum('bik,bk->bi', v[ids], u[uid])
            cos = dot/(np.linalg.norm(v[ids], axis=2)*np.linalg.norm(u[uid], axis=1)[:, None]).clip(1e-6)
            p = predict(ema, raw, valid, device)
            row = dict(epoch=epoch, dot_auc=float(roc_auc_score(labels.ravel(), dot.ravel())),
                       cosine_auc=float(roc_auc_score(labels.ravel(), cos.ravel())),
                       event_model_auc=float(roc_auc_score(labels.ravel(), p.ravel())), elapsed=time.monotonic()-started)
            records.append(row)
            if epoch in [0, 4, 16]:
                torch.save(dict(state_dict=ema.state_dict(), config=config), out/('model_e'+str(epoch)+'.pt'))
            write_json(out/'report.json', report)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', default='2025-01-26', choices=['2025-01-26', '2025-01-25'])
    run(parser.parse_args().cutoff)
