"""Checks for leakage, unknown IDs, empty histories and submission alignment."""

import contextlib
import argparse
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ctr_data import BASE_COLUMNS, fit_priors, load_cache, prepare, temporal_split, with_history_limit
from ctr_model import HistoryTwoTower
from light import run_light
from light_data import build_light_dataset
from run import check_submission, get_batch, predict, retrain, timestamp, train


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cutoff = timestamp("2025-01-26")
        self.train_path = self.root / "train.csv"
        self.eval_path = self.root / "evaluation.csv"
        train_events = [
            ("a", self.cutoff-172800, [], [], 1),
            ("b", self.cutoff-86400, ["warmup"], [self.cutoff-200000], 0),
            ("a", self.cutoff, ["item0"], [self.cutoff-172800], 1),
            ("a", self.cutoff+3600, ["item0"], [self.cutoff-100], 0),
            ("new", self.cutoff+7200, ["future_only"], [self.cutoff+6000], 1),
        ]
        self.write_events(self.train_path, train_events, True)
        self.write_events(self.eval_path, [
            ("unseen_eval", self.cutoff+10000, [], [], 0),
            ("a", self.cutoff+11000, ["item0"], [self.cutoff], 0),
        ], False)
        self.cache = self.root / "cache"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.cache, self.train_path, self.eval_path, train_events=5, evaluation_events=2)
        self.meta, self.data = load_cache(self.cache, "train")

    def tearDown(self):
        self.temp.cleanup()

    def write_events(self, path, events, labelled):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(BASE_COLUMNS + (["clicked"] if labelled else []))
            for e, (uid, ts, history, times, clicked) in enumerate(events):
                for slot in range(10):
                    row = [f"{path.stem}_row_{e}_{slot}", f"event_{e}", uid, f"item{slot}", ts,
                           (ts % 86400)//21600, 1, json.dumps(history), json.dumps(times)]
                    writer.writerow(row + ([int(clicked and slot == 0)] if labelled else []))

    def test_split_is_temporal_and_keeps_only_first_event(self):
        fit, valid = temporal_split(self.data, self.cutoff)
        np.testing.assert_array_equal(fit, [0, 1])
        np.testing.assert_array_equal(valid, [2, 4])
        self.assertEqual(self.data["items"].shape, (5, 10))
        self.assertTrue((self.data["timestamps"][fit] < self.cutoff).all())

    def test_validation_labels_and_future_ids_do_not_affect_priors(self):
        fit, valid = temporal_split(self.data, self.cutoff)
        before = fit_priors(self.data, fit, self.meta["n_users"], self.meta["n_items"])
        altered = dict(self.data)
        altered["labels"] = np.array(self.data["labels"])
        altered["labels"][valid] = 1
        after = fit_priors(altered, fit, self.meta["n_users"], self.meta["n_items"])
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)
        vocab = json.loads((self.cache / "vocabulary.json").read_text())
        self.assertFalse(before[3][vocab["users"]["new"]])
        self.assertFalse(before[4][vocab["items"]["future_only"]])
        self.assertTrue(before[4][vocab["items"]["warmup"]])

    def test_model_handles_unknown_user_empty_history_and_zero_click_event(self):
        fit, _ = temporal_split(self.data, self.cutoff)
        model = HistoryTwoTower(self.meta["n_users"], self.meta["n_items"], dim=8)
        model.initialize_priors(*fit_priors(self.data, fit, self.meta["n_users"], self.meta["n_items"]))
        model.eval()
        _, evaluation = load_cache(self.cache, "evaluation")
        batch = get_batch(evaluation, np.arange(2), torch.device("cpu"))
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(torch.isfinite(logits).all())
        loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        path = self.root / "weights.pt"
        torch.save(model.state_dict(), path)
        clone = HistoryTwoTower(self.meta["n_users"], self.meta["n_items"], dim=8).eval()
        clone.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(clone(batch), logits)

    def test_checker_matches_by_rowid_and_rejects_invalid_files(self):
        with self.eval_path.open() as handle:
            rowids = [r["rowid"] for r in csv.DictReader(handle)]
        path = self.root / "submission.csv"

        def write(rows):
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["rowid", "predicted_probability"])
                writer.writerows(rows)

        write([(rid, 0.1) for rid in reversed(rowids)])
        with contextlib.redirect_stdout(io.StringIO()):
            check_submission(path, self.eval_path)
        for bad_rows in (
            [(rid, 0.1) for rid in rowids[:-1]],
            [(rid, 0.1) for rid in rowids] + [(rowids[0], 0.1)],
            [(rid, "nan" if i == 0 else 0.1) for i, rid in enumerate(rowids)],
            [("unknown" if i == 0 else rid, 0.1) for i, rid in enumerate(rowids)],
        ):
            write(bad_rows)
            with self.assertRaises(ValueError):
                check_submission(path, self.eval_path)

    def test_validation_retraining_and_submission_roundtrip(self):
        validation = self.root / "validation.pt"
        full = self.root / "full.pt"
        output = self.root / "predictions.csv"
        args = argparse.Namespace(
            command="train", cache=str(self.cache), checkpoint=str(validation), cutoff="2025-01-26",
            full=False, epochs=1, dim=8, batch_size=2, lr=0.002, weight_decay=0.00001,
            user_dropout=0.1, patience=2, seed=42, max_train_events=0, max_valid_events=0,
            device="cpu", threads=1,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
            prediction_args = argparse.Namespace(
                cache=str(self.cache), checkpoint=str(validation), out=str(output),
                evaluation=str(self.eval_path), batch_size=2, device="cpu", threads=1, allow_partial=False,
            )
            with self.assertRaisesRegex(ValueError, "full-training checkpoint"):
                predict(prediction_args)
            retrain(argparse.Namespace(cache=str(self.cache), validation_checkpoint=str(validation),
                                       checkpoint=str(full), device="cpu", threads=1))
            prediction_args.checkpoint = str(full)
            predict(prediction_args)
        saved = torch.load(full, weights_only=True)
        self.assertTrue(saved["full_training"])
        self.assertFalse(saved["smoke_test"])
        self.assertEqual(saved["epoch"], 1)
        with output.open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 20)

    def test_history_limit_updates_all_history_inputs_without_mutating_cache(self):
        data = {k: np.array(v) for k, v in self.data.items()}
        data["history"][4, :3] = data["items"][4, :3]
        data["age_hours"][4, :3] = [72, 48, 24]
        limited = with_history_limit(data, 2)
        self.assertEqual(limited["history"][4, 0], 0)
        np.testing.assert_array_equal(limited["history"][4, 1:3], data["items"][4, 1:3])
        self.assertGreater(data["history"][4, 0], 0)
        self.assertAlmostEqual(float(limited["context"][4, 3]), 2/20)
        self.assertAlmostEqual(float(limited["context"][4, 4]), np.log1p(24)/np.log1p(720))
        self.assertAlmostEqual(float(limited["context"][4, 5]), np.log1p(24)/np.log1p(720))
        empty = with_history_limit(data, 0)
        self.assertFalse(empty["history"].any())
        self.assertFalse(empty["age_hours"].any())
        self.assertFalse(empty["context"][:, 3:].any())

    def test_light_sampling_keeps_validation_fixed_when_training_volume_changes(self):
        # Give the selected returning user two fitting events so subsampling
        # actually changes the training set in this tiny fixture.
        users = np.load(self.cache / "train/users.npy", mmap_mode="r+")
        users[1] = users[0]
        users.flush()
        first = build_light_dataset(self.cache, self.cutoff, cohort_users=2, split_seed=7)
        repeated = build_light_dataset(self.cache, self.cutoff, cohort_users=2, split_seed=7)
        smaller = build_light_dataset(self.cache, self.cutoff, cohort_users=2, split_seed=7, fit_fraction=0.5)
        np.testing.assert_array_equal(first[2], repeated[2])
        self.assertEqual(first[3]["split_id"], repeated[3]["split_id"])
        np.testing.assert_array_equal(first[2][first[1][1]], smaller[2][smaller[1][1]])
        self.assertEqual(first[3]["validation_id"], smaller[3]["validation_id"])
        self.assertNotEqual(first[3]["split_id"], smaller[3]["split_id"])
        self.assertLess(smaller[3]["fit_events"], first[3]["fit_events"])
        self.assertTrue((first[0]["timestamps"][first[1][0]] < self.cutoff).all())
        self.assertTrue((first[0]["timestamps"][first[1][1]] >= self.cutoff).all())

    def test_light_chain_exports_baselines_checkpoint_and_checked_submission(self):
        config = {
            "cache": str(self.cache), "evaluation": str(self.eval_path), "cutoff": "2025-01-26",
            "cohort_users": 2, "split_seed": 7, "fit_fraction": 1.0, "history_limit": 0,
            "dim": 4, "epochs": 2, "lr": 0.002, "weight_decay": 0.01, "user_dropout": 0.2,
            "batch_size": 2, "seed": 42, "device": "cpu", "threads": 1,
            "use_history_pooling": False, "use_repeat_features": False, "predict": True,
        }
        output = self.root / "light"
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_light(config, output)
        self.assertEqual(summary["baselines"]["constant"]["auc"], 0.5)
        self.assertTrue(summary["submission_checked"])
        self.assertTrue(summary["subset_experiment"])
        saved = torch.load(output / "model.pt", weights_only=True)
        self.assertEqual(saved["history_limit"], 0)
        self.assertFalse(saved["config"]["use_history_pooling"])
        self.assertFalse(saved["config"]["use_repeat_features"])
        with (output / "metrics.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)
        predictions = np.load(output / "model.validation.npz")
        original = np.load(output / "split.npz")
        np.testing.assert_array_equal(predictions["event_indices"], original["validation_indices"])
        np.testing.assert_array_equal(predictions["labels"], self.data["labels"][original["validation_indices"]])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
