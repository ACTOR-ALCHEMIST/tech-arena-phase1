#!/usr/bin/env python3
"""Run prepare -> train -> retrain -> predict -> check. See --help."""

import argparse
import csv
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.nn import functional as F

from ctr_data import csv_reader, fit_priors, load_cache, prepare, temporal_split, with_history_limit
from ctr_model import HistoryTwoTower


def timestamp(date):
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this Python environment")
    return torch.device(name)


def get_batch(data, indices, device, labels=False):
    result = {}
    for key in ("users", "items", "history", "age_hours", "context"):
        dtype = torch.long if key in ("users", "items", "history") else torch.float32
        result[key] = torch.as_tensor(np.array(data[key][indices]), dtype=dtype, device=device)
    if labels:
        result["labels"] = torch.as_tensor(np.array(data["labels"][indices]), dtype=torch.float32, device=device)
    return result


@torch.inference_mode()
def predict_arrays(model, data, indices, batch_size, device):
    model.eval()
    pieces = []
    for start in range(0, len(indices), batch_size):
        batch = get_batch(data, indices[start:start+batch_size], device)
        pieces.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(pieces)


def metrics(labels, predictions):
    y, p = labels.ravel(), predictions.ravel()
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite predictions")
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1]))}


def report_slices(data, indices, predictions, known_users):
    groups = {
        "known_user": known_users[data["users"][indices]],
        "new_user": ~known_users[data["users"][indices]],
        "empty_history": (data["history"][indices] != 0).sum(1) == 0,
    }
    result = {}
    for name, mask in groups.items():
        labels = data["labels"][indices[mask]]
        result[name] = {"events": int(mask.sum())}
        if labels.size and len(np.unique(labels)) == 2:
            result[name].update(metrics(labels, predictions[mask]))
    return result


def train(args, data_override=None, split_override=None, source_indices=None):
    if args.epochs < 1 or args.batch_size < 1 or args.dim < 1:
        raise ValueError("epochs, batch-size and dim must be positive")
    if args.full and (args.max_train_events or args.max_valid_events):
        raise ValueError("Full training cannot use smoke-test limits")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    meta, data = load_cache(args.cache, "train")
    if data_override is not None:
        data = data_override
    history_limit = getattr(args, "history_limit", 20)
    data = with_history_limit(data, history_limit)
    if split_override is not None:
        if args.full:
            raise ValueError("A subset split cannot be used for full training")
        fit, valid = split_override
    elif args.full:
        fit = np.arange(len(data["users"]))
        valid = np.empty(0, dtype=np.int64)
    else:
        fit, valid = temporal_split(data, timestamp(args.cutoff))
    rng = np.random.default_rng(args.seed)
    # A random sample across the fitting period is more representative than its first rows.
    if args.max_train_events and len(fit) > args.max_train_events:
        fit = np.sort(rng.choice(fit, args.max_train_events, replace=False))
    if args.max_valid_events and len(valid) > args.max_valid_events:
        valid = np.sort(rng.choice(valid, args.max_valid_events, replace=False))
    smoke = bool(args.max_train_events or args.max_valid_events or split_override is not None)
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        raise ValueError(f"Checkpoint exists: {checkpoint}. Choose a new name to preserve the experiment.")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ctr, ub, ib, known_users, known_items = fit_priors(data, fit, meta["n_users"], meta["n_items"])
    config = {"n_users": meta["n_users"], "n_items": meta["n_items"], "dim": args.dim,
              "user_dropout": args.user_dropout,
              "use_history_pooling": getattr(args, "use_history_pooling", True),
              "use_repeat_features": getattr(args, "use_repeat_features", True)}
    model = HistoryTwoTower(**config)
    model.initialize_priors(ctr, ub, ib, known_users, known_items)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    report = {"cache_id": meta["cache_id"], "arguments": vars(args), "config": config,
              "fit_events": len(fit), "validation_events": len(valid), "smoke_test": smoke,
              "device": str(device), "fit_ctr": ctr, "epochs": []}
    print(json.dumps({k: report[k] for k in ("fit_events", "validation_events", "smoke_test", "device", "fit_ctr")}), flush=True)
    if len(valid):
        report["constant_baseline"] = metrics(data["labels"][valid], np.full((len(valid), 10), ctr))
        print("Constant probability baseline:", report["constant_baseline"], flush=True)
        baseline_logits = np.log(ctr / (1-ctr)) + ub[data["users"][valid]][:, None] + ib[data["items"][valid]]
        baseline = 1 / (1 + np.exp(-np.clip(baseline_logits, -50, 50)))
        report["prior_baseline"] = metrics(data["labels"][valid], baseline)
        print("Smoothed user/item CTR baseline:", report["prior_baseline"], flush=True)
    best, best_epoch, stale = -float("inf"), 0, 0
    for epoch in range(1, args.epochs+1):
        started = time.monotonic()
        model.train()
        order = rng.permutation(fit)
        total_loss, seen = 0.0, 0
        for step, start in enumerate(range(0, len(order), args.batch_size), 1):
            chosen = order[start:start+args.batch_size]
            batch = get_batch(data, chosen, device, labels=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(batch), batch["labels"])
            if not torch.isfinite(loss):
                raise ValueError("Training loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(chosen)
            seen += len(chosen)
            if step % 200 == 0:
                print(f"epoch {epoch}, {seen:,}/{len(fit):,} events, loss={total_loss/seen:.5f}", flush=True)
        row = {"epoch": epoch, "train_logloss": total_loss/seen}
        predictions = None
        if len(valid):
            predictions = predict_arrays(model, data, valid, args.batch_size, device)
            row.update(metrics(data["labels"][valid], predictions))
            row["auc_gain_vs_prior"] = row["auc"] - report["prior_baseline"]["auc"]
        row["seconds"] = round(time.monotonic()-started, 2)
        report["epochs"].append(row)
        print(json.dumps(row), flush=True)
        score = row.get("auc", float(epoch))
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config, "cache_id": meta["cache_id"],
                        "epoch": epoch, "full_training": args.full, "smoke_test": smoke,
                        "seed": args.seed, "metrics": row, "history_limit": history_limit}, checkpoint)
            if predictions is not None:
                np.savez_compressed(checkpoint.with_suffix(".validation.npz"),
                                    event_indices=source_indices[valid] if source_indices is not None else valid,
                                    labels=data["labels"][valid], predictions=predictions,
                                    prior_baseline=baseline)
                report["best_slices"] = report_slices(data, valid, predictions, known_users)
        else:
            stale += 1
        report["best_epoch"] = best_epoch
        checkpoint.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if len(valid) and stale >= args.patience:
            print("Early stopping: validation AUC did not improve", flush=True)
            break
    print(f"Saved {checkpoint}; best_epoch={best_epoch}" + (f"; best_auc={best:.6f}" if len(valid) else ""), flush=True)
    if smoke:
        print("SUBSET EXPERIMENT: confirm promising changes on the full temporal split before competition decisions.", flush=True)
    return report


def predict(args):
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    meta, data = load_cache(args.cache, "evaluation")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["cache_id"] != meta["cache_id"]:
        raise ValueError("Checkpoint and cache do not match")
    data = with_history_limit(data, saved.get("history_limit", 20))
    if not saved["full_training"] or saved["smoke_test"]:
        if not args.allow_partial:
            raise ValueError("Use a full-training checkpoint for submission. --allow-partial is only for pipeline testing.")
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Output exists: {out}; choose a new filename")
    out.parent.mkdir(parents=True, exist_ok=True)
    model = HistoryTwoTower(**saved["config"])
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    indices = np.arange(len(data["users"]))
    probabilities = predict_arrays(model, data, indices, args.batch_size, device)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid predicted probabilities")
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rowid", "predicted_probability"])
        for rowids, probs in zip(data["rowids"], probabilities):
            writer.writerows((rid.decode("utf-8"), f"{float(p):.9g}") for rid, p in zip(rowids, probs))
    check_submission(out, args.evaluation)
    print(f"Saved {out} using epoch {saved['epoch']} ({device})", flush=True)


def retrain(args):
    """Start fresh on all training events, using the validation-selected epoch count."""
    validation = Path(args.validation_checkpoint)
    if not validation.is_file():
        raise ValueError("Run a full-size validation experiment first")
    report = json.loads(validation.with_suffix(".json").read_text())
    if report["smoke_test"] or report["arguments"]["full"]:
        raise ValueError("Epoch selection requires a full-size validation experiment")
    meta, _ = load_cache(args.cache, "train")
    if report["cache_id"] != meta["cache_id"]:
        raise ValueError("Validation experiment and cache do not match")
    values = dict(report["arguments"])
    values.update(full=True, epochs=report["best_epoch"], checkpoint=args.checkpoint,
                  cache=args.cache, device=args.device, threads=args.threads)
    print(f"Retraining from scratch on all events for {values['epochs']} epochs", flush=True)
    train(argparse.Namespace(**values))


def check_submission(submission, evaluation):
    expected = set()
    with csv_reader(evaluation) as reader:
        header = next(reader)
        index = header.index("rowid")
        for row in reader:
            if row[index] in expected:
                raise ValueError("Duplicate rowid in evaluation source")
            expected.add(row[index])
    seen = set()
    low, high = 1.0, 0.0
    with Path(submission).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["rowid", "predicted_probability"]:
            raise ValueError("Incorrect submission header")
        for line, row in enumerate(reader, 2):
            if len(row) != 2:
                raise ValueError(f"Incorrect column count at line {line}")
            rid, raw = row
            if rid in seen or rid not in expected:
                raise ValueError(f"Duplicate/unknown rowid at line {line}: {rid}")
            if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", raw):
                raise ValueError(f"Invalid numeric format at line {line}")
            try:
                p = float(raw)
            except ValueError as exc:
                raise ValueError(f"Nonnumeric probability at line {line}") from exc
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError(f"Invalid probability at line {line}")
            seen.add(rid)
            low, high = min(low, p), max(high, p)
    if seen != expected:
        raise ValueError(f"Submission is missing {len(expected-seen)} evaluation rows")
    print(f"Submission checks passed: {len(seen):,} rows; probability range [{low:.6g}, {high:.6g}]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Stream CSV/ZIP files into compact event-level arrays")
    p.add_argument("--cache", default="data/cache_v1")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--evaluation", default="evaluation.zip")
    p = commands.add_parser("train", help="Train and validate; --full fits all training events")
    p.add_argument("--cache", default="data/cache_v1")
    p.add_argument("--checkpoint", default="model/two_tower_val.pt")
    p.add_argument("--cutoff", default="2025-01-26", help="UTC date; ignored for --full")
    p.add_argument("--full", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=512, help="Events per batch; each event has 10 candidates")
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.00001)
    p.add_argument("--user-dropout", type=float, default=0.1)
    p.add_argument("--history-limit", type=int, default=20, help="Keep the most recent 0..20 clicks")
    p.add_argument("--no-history-pooling", dest="use_history_pooling", action="store_false")
    p.add_argument("--no-repeat-features", dest="use_repeat_features", action="store_false")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-events", type=int, default=0, help="Smoke test only; zero uses the entire fitting period")
    p.add_argument("--max-valid-events", type=int, default=0, help="Smoke test only")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--threads", type=int, default=4)
    p = commands.add_parser("retrain", help="Fit all training events using the validation-selected epoch count")
    p.add_argument("--cache", default="data/cache_v1")
    p.add_argument("--validation-checkpoint", default="model/two_tower_val.pt")
    p.add_argument("--checkpoint", default="model/two_tower_full.pt")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--threads", type=int, default=4)
    p = commands.add_parser("predict", help="Generate a submission and check it against evaluation rowids")
    p.add_argument("--cache", default="data/cache_v1")
    p.add_argument("--checkpoint", default="model/two_tower_full.pt")
    p.add_argument("--out", default="submissions/two_tower_v1.csv")
    p.add_argument("--evaluation", default="evaluation.zip")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--allow-partial", action="store_true", help="Allow a smoke/validation model for testing only")
    p = commands.add_parser("check", help="Check CSV format, full rowid coverage and finite probabilities")
    p.add_argument("submission")
    p.add_argument("--evaluation", default="evaluation.zip")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.cache, args.train, args.evaluation)
    elif args.command == "train":
        train(args)
    elif args.command == "retrain":
        retrain(args)
    elif args.command == "predict":
        predict(args)
    else:
        check_submission(args.submission, args.evaluation)


if __name__ == "__main__":
    main()
