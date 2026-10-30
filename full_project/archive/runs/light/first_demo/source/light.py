#!/usr/bin/env python3
"""One-command small-cohort baseline -> training -> prediction -> CSV checks."""

import argparse
import csv
import json
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from light_data import build_light_dataset
from run import predict, timestamp, train


def run_light(config, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Each run remains reviewable after editing data/model code for the next run.
    sources = output / "source"
    sources.mkdir()
    for name in ("light.py", "light_data.py", "run.py", "ctr_data.py", "ctr_model.py", "requirements.txt"):
        shutil.copy2(Path(__file__).parent / name, sources / name)
    print("[1/4] Prepare a fixed small user cohort", flush=True)
    data, split, source_indices, dataset = build_light_dataset(
        config["cache"], timestamp(config["cutoff"]), config["cohort_users"], config["split_seed"])
    print(json.dumps(dataset, indent=2), flush=True)
    np.savez_compressed(output / "split.npz", fit_indices=source_indices[split[0]],
                        validation_indices=source_indices[split[1]])
    print("[2/4] Evaluate baselines and train a small model", flush=True)
    checkpoint = output / "model.pt"
    args = argparse.Namespace(
        command="train", cache=config["cache"], checkpoint=str(checkpoint), cutoff=config["cutoff"],
        full=False, epochs=config["epochs"], dim=config["dim"], batch_size=config["batch_size"],
        lr=config["lr"], weight_decay=config["weight_decay"], user_dropout=config["user_dropout"],
        history_limit=config["history_limit"], use_history_pooling=config["use_history_pooling"],
        use_repeat_features=config["use_repeat_features"], patience=config["epochs"], seed=config["seed"],
        max_train_events=0, max_valid_events=0, device=config["device"], threads=config["threads"],
    )
    report = train(args, data_override=data, split_override=split, source_indices=source_indices)
    best = max(report["epochs"], key=lambda row: row["auc"])
    rows = [
        {"stage": "constant", **report["constant_baseline"], "auc_gain_vs_prior":
         report["constant_baseline"]["auc"]-report["prior_baseline"]["auc"]},
        {"stage": "user_item_ctr", **report["prior_baseline"], "auc_gain_vs_prior": 0.0},
    ] + [{"stage": f"epoch_{row['epoch']}", **row} for row in report["epochs"]]
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "epoch", "auc", "logloss", "auc_gain_vs_prior", "seconds"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    submission = None
    if config["predict"]:
        print("[3/4] Reload the best checkpoint and predict ALL evaluation rows", flush=True)
        submission = output / "submission_test.csv"
        predict(argparse.Namespace(
            cache=config["cache"], checkpoint=str(checkpoint), out=str(submission),
            evaluation=config.get("evaluation", "evaluation.zip"), batch_size=1024,
            device=config["device"], threads=config["threads"], allow_partial=True,
        ))
    else:
        print("[3/4] Prediction skipped for faster iteration", flush=True)
    summary = {
        **dataset, "config": config, "baselines": {"constant": report["constant_baseline"],
        "user_item_ctr": report["prior_baseline"]}, "best_epoch": best["epoch"],
        "best_auc": best["auc"], "auc_gain_vs_prior": best["auc_gain_vs_prior"],
        "epochs": report["epochs"], "slices": report["best_slices"],
        "submission_test": str(submission) if submission else None,
        "submission_checked": submission is not None, "subset_experiment": True,
        "elapsed_seconds": round(time.monotonic()-started, 2),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[4/4] Results (same validation rows for every stage)", flush=True)
    print(f"  constant       AUC {report['constant_baseline']['auc']:.6f}", flush=True)
    print(f"  user/item CTR  AUC {report['prior_baseline']['auc']:.6f}", flush=True)
    for row in report["epochs"]:
        print(f"  epoch {row['epoch']:<3}      AUC {row['auc']:.6f}  delta {row['auc_gain_vs_prior']:+.6f}", flush=True)
    print(f"  best epoch {best['epoch']}; elapsed {summary['elapsed_seconds']}s; output: {output}", flush=True)
    print("Small-cohort results guide experiments; confirm improvements on the full temporal split.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/light.json")
    parser.add_argument("--name", help="Optional run name; existing runs are preserved")
    parser.add_argument("--skip-predict", action="store_true", help="Skip evaluation CSV generation on later experiments")
    parser.add_argument("--history-limit", type=int, help="Data experiment: keep 0..20 recent clicks")
    parser.add_argument("--dim", type=int, help="Model experiment: embedding dimension")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, help="Model seed; the cohort still uses split_seed")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ("history_limit", "dim", "epochs", "seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.skip_predict:
        config["predict"] = False
    if not 0 <= config["history_limit"] <= 20:
        parser.error("history-limit must be between 0 and 20")
    if config["epochs"] < 1 or config["dim"] < 1:
        parser.error("epochs and dim must be positive")
    name = args.name or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        parser.error("name must contain only letters, digits, dots, underscores or hyphens")
    run_light(config, Path("runs/light") / name)


if __name__ == "__main__":
    main()
