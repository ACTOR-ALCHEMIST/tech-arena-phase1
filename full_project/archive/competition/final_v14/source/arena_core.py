"""Time-safe sparse collaborative filtering and reusable numeric features."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".tools/python"))
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import fcntl
import json
import time

import numpy as np
from numba import njit, prange, set_num_threads
from scipy import sparse
from sklearn.metrics import average_precision_score, accuracy_score, log_loss, roc_auc_score

from ctr_data import fit_priors, load_cache, temporal_split
from run import timestamp
from arena_report import LAB, note, render

set_num_threads(6)


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -40, 40)))


def evaluate(labels, predictions):
    y, p = np.asarray(labels).ravel(), np.asarray(predictions).ravel()
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite predictions")
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "accuracy_at_0_5": float(accuracy_score(y, p >= 0.5)), "rows": len(y)}


def snapshot(cutoff, cache="data/cache_v1"):
    parent=Path(os.environ.get("ARENA_SNAPSHOT_ROOT",str(LAB/"cache")))
    path=parent/("snapshot_"+cutoff)
    path.mkdir(parents=True,exist_ok=True)
    with (path/".build.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _snapshot_unlocked(cutoff,cache)


def _snapshot_unlocked(cutoff, cache="data/cache_v1"):
    path = Path(os.environ.get("ARENA_SNAPSHOT_ROOT",str(LAB/"cache"))) / ("snapshot_"+cutoff)
    meta, data = load_cache(cache, "train")
    if (path / "metadata.json").exists():
        info = json.loads((path / "metadata.json").read_text())
        if info["cache_id"] != meta["cache_id"]:
            raise ValueError("Snapshot and data cache differ")
        return path
    path.mkdir(parents=True, exist_ok=True)
    fit = np.flatnonzero(data["timestamps"] < timestamp(cutoff))
    if not len(fit):
        raise ValueError("No fitting events before cutoff")
    start = time.monotonic()
    h = np.array(data["history"][fit]).ravel()
    u = np.repeat(data["users"][fit], 20)
    keep = h != 0
    labels = data["labels"][fit]
    rows, slots = np.nonzero(labels)
    ui_users = np.concatenate([u[keep], data["users"][fit[rows]]])
    ui_items = np.concatenate([h[keep], data["items"][fit[rows], slots]])
    x = sparse.coo_matrix((np.ones(len(ui_users), dtype=np.float32), (ui_users, ui_items)),
                          shape=(meta["n_users"], meta["n_items"])).tocsr()
    x.data.fill(1)
    x.sort_indices()
    del ui_users, ui_items, u, h, keep
    sparse.save_npz(path / "interactions.npz", x)
    pop = np.asarray(x.sum(axis=0)).ravel()
    print(f"snapshot {cutoff}: {len(fit):,} fitting events; {x.nnz:,} unique user/item positives", flush=True)
    counts = (x.T @ x).tocsr()
    counts.setdiag(0)
    counts.eliminate_zeros()
    counts.sort_indices()
    sparse.save_npz(path / "co_counts.npz", counts)
    ctr, ub, ib, known_u, known_i = fit_priors(data, fit, meta["n_users"], meta["n_items"])
    uc = np.bincount(data["users"][fit], weights=labels.sum(1), minlength=meta["n_users"])
    ui = np.bincount(data["users"][fit], minlength=meta["n_users"]) * 10
    ic = np.bincount(data["items"][fit].ravel(), weights=labels.ravel(), minlength=meta["n_items"])
    ii = np.bincount(data["items"][fit].ravel(), minlength=meta["n_items"])
    np.savez(path / "stats.npz", ctr=ctr, ub=ub, ib=ib, pop=pop, uc=uc, ui=ui, ic=ic, ii=ii,
             known_users=known_u, known_items=known_i)
    info = {"cache_id": meta["cache_id"], "cutoff": cutoff, "fit_events": len(fit),
            "unique_interactions": x.nnz, "co_nnz": counts.nnz, "seconds": time.monotonic()-start,
            "max_fit_timestamp": int(data["timestamps"][fit].max()), "cutoff_timestamp": timestamp(cutoff)}
    (path / "metadata.json").write_text(json.dumps(info, indent=2))
    note(f"构建 {cutoff} 严格前向快照：{len(fit):,} 个事件、{x.nnz:,} 个去重用户物品交互、{counts.nnz:,} 个共点击非零项，耗时 {info['seconds']:.1f} 秒。")
    return path


@njit(parallel=True, cache=True)
def co_features(indptr, columns, values, pop, candidates, histories, ages):
    n = len(candidates)
    out = np.zeros((n, 10, 9), dtype=np.float32)
    inv = 1/np.sqrt(np.maximum(pop, 1))
    for e in prange(n):
        for j in range(10):
            item = candidates[e, j]
            a, b = indptr[item], indptr[item+1]
            for k in range(20):
                h = histories[e, k]
                if h == 0:
                    continue
                lo, hi = a, b
                while lo < hi:
                    mid = (lo+hi)//2
                    if columns[mid] < h:
                        lo = mid+1
                    else:
                        hi = mid
                if lo >= b or columns[lo] != h:
                    continue
                count = values[lo]
                cosine = count*inv[item]*inv[h]
                age_days = ages[e, k]/24
                out[e, j, 0] += cosine
                out[e, j, 1] = max(out[e, j, 1], cosine)
                out[e, j, 2] += cosine*np.exp(-age_days/0.5)
                out[e, j, 3] += cosine*np.exp(-age_days/3)
                out[e, j, 4] += cosine*np.exp(-age_days/30)
                out[e, j, 5] += cosine*count/(count+3)
                out[e, j, 6] += np.log1p(count)
                out[e, j, 7] += count/max(pop[h], 1)
                if k == 19 or histories[e, k+1] == 0:
                    out[e, j, 8] = cosine
    return out


def load_snapshot(path):
    return sparse.load_npz(path / "co_counts.npz"), dict(np.load(path / "stats.npz"))


def features(data, indices, path):
    c, stats = load_snapshot(path)
    items = np.array(data["items"][indices])
    users = np.array(data["users"][indices])
    h = np.array(data["history"][indices])
    ages = np.array(data["age_hours"][indices])
    cf = co_features(c.indptr, c.indices, c.data, stats["pop"].astype(np.float32), items, h, ages)
    del c
    names = ["cf_cos_sum", "cf_cos_max", "cf_recent_12h", "cf_recent_3d", "cf_recent_30d",
             "cf_shrink3", "cf_log_count", "cf_conditional", "cf_last"]
    parts = [cf]
    def add(name, value):
        names.append(name)
        if value.ndim == 1:
            value = np.broadcast_to(value[:, None], items.shape)
        parts.append(value[..., None].astype(np.float32))
    prior = float(np.log(stats["ctr"]/(1-stats["ctr"])))+stats["ub"][users, None]+stats["ib"][items]
    add("prior_logit", prior)
    add("user_bias", stats["ub"][users]);add("item_bias", stats["ib"][items])
    for name in ("uc", "ui"):
        add("log_"+name, np.log1p(stats[name][users]))
    for name in ("ic", "ii", "pop"):
        add("log_"+name, np.log1p(stats[name][items]))
    mask = (items[:, :, None] == h[:, None, :]) & (h[:, None, :] != 0)
    count = mask.sum(2).astype(np.float32)
    add("repeat_any", (count>0).astype(np.float32));add("repeat_count", count)
    for tau in (0.5, 3, 30):
        add("repeat_decay_"+str(tau), (mask*np.exp(-ages[:, None, :]/(24*tau))).sum(2))
    for k, name in enumerate(("hour_sin", "hour_cos", "weekend", "history_length", "last_gap", "history_span")):
        add(name, np.array(data["context"][indices, k]))
    for k in (0, 1, 3, 5):
        add(names[k]+"_relative", cf[:, :, k]-cf[:, :, k].mean(1, keepdims=True))
        add(names[k]+"_rank", np.argsort(np.argsort(cf[:, :, k], axis=1), axis=1).astype(np.float32)/9)
    add("item_bias_relative", stats["ib"][items]-stats["ib"][items].mean(1, keepdims=True))
    matrix = np.concatenate(parts, axis=2).reshape(-1, len(names))
    return matrix, names, prior, cf


def cf_benchmark(cutoff, name):
    start = time.monotonic()
    path = snapshot(cutoff)
    _, data = load_cache("data/cache_v1", "train")
    _, valid = temporal_split(data, timestamp(cutoff))
    x, names, prior, cf = features(data, valid, path)
    candidates = {}
    for weight in (0, 5, 10, 25, 50, 100):
        candidates[f"starter_cf_w{weight}"] = sigmoid(prior+weight*cf[:, :, 0])
    for col in (0, 1, 2, 3, 4, 5, 7):
        for weight in (10, 25, 50):
            candidates[f"cf_f{col}_w{weight}"] = sigmoid(prior+weight*cf[:, :, col])
    scores = {key: evaluate(data["labels"][valid], p) for key, p in candidates.items()}
    best = max(scores, key=lambda key: scores[key]["auc"])
    out = LAB / "experiments" / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "validation.npz", event_indices=valid, labels=data["labels"][valid],
                        predictions=candidates[best], starter=candidates["starter_cf_w25"])
    np.save(out / "features.npy", x)
    (out / "feature_names.json").write_text(json.dumps(names))
    result = {"name": name, "cutoff": cutoff, "method": "sparse_item_cf", "best_variant": best,
              **scores[best], "seconds": time.monotonic()-start, "all_variants": scores,
              "snapshot": str(path.relative_to(ROOT)), "validation_events": len(valid)}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    note(f"实验 {name}：AUC {result['auc']:.6f}，官方三信号公式 AUC {scores['starter_cf_w25']['auc']:.6f}，最佳 {best}。", "experiment")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("command", choices=["snapshot", "cf"]);p.add_argument("--cutoff", default="2025-01-26");p.add_argument("--name", default="cf_26")
    a=p.parse_args()
    if a.command == "snapshot":
        print(snapshot(a.cutoff))
    else:
        cf_benchmark(a.cutoff, a.name)
