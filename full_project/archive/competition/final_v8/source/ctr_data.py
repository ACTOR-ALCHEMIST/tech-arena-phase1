"""Step 1: stream the competition CSVs into compact, event-level NumPy arrays."""

import csv
import io
import itertools
import json
import math
import uuid
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np


BASE_COLUMNS = [
    "rowid", "event_id", "user_id", "item_id", "timestamp", "time_bucket",
    "day_type", "history_item_ids", "history_timestamps",
]
ARRAYS = ("users", "items", "timestamps", "history", "age_hours", "context")


@contextmanager
def csv_reader(path):
    """Use the CSV parser, including when reading directly from a ZIP archive."""
    path = Path(path)
    with ExitStack() as stack:
        if path.suffix.lower() == ".zip":
            archive = stack.enter_context(zipfile.ZipFile(path))
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one CSV in {path}")
            raw = stack.enter_context(archive.open(names[0]))
            handle = stack.enter_context(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        else:
            handle = stack.enter_context(path.open(encoding="utf-8", newline=""))
        yield csv.reader(handle)


def _source(path):
    p = Path(path).resolve()
    stat = p.stat()
    return {"path": str(p), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare(cache, train="train.csv", evaluation="evaluation.zip",
            train_events=686_824, evaluation_events=52_939):
    cache = Path(cache)
    sources = {"train": _source(train), "evaluation": _source(evaluation)}
    if (cache / "metadata.json").exists():
        meta = json.loads((cache / "metadata.json").read_text())
        if meta["sources"] != sources:
            raise ValueError("Sources changed. Choose a new --cache directory.")
        print(f"Cache already prepared: {cache}", flush=True)
        return meta
    if cache.exists() and any(cache.iterdir()):
        raise ValueError(f"Incomplete/nonempty cache: {cache}. Choose a new --cache directory.")
    cache.mkdir(parents=True, exist_ok=True)
    # Token 0 is reserved for unknown IDs and history padding.
    users, items = {}, {}

    def encode(mapping, value):
        if value not in mapping:
            mapping[value] = len(mapping) + 1
        return mapping[value]

    summaries = {}
    for name, source, expected, labelled in (
        ("train", train, train_events, True),
        ("evaluation", evaluation, evaluation_events, False),
    ):
        folder = cache / name
        folder.mkdir()
        specs = {
            "users": ("int32", (expected,)),
            "items": ("int32", (expected, 10)),
            "timestamps": ("int64", (expected,)),
            "history": ("int32", (expected, 20)),
            "age_hours": ("float32", (expected, 20)),
            "context": ("float32", (expected, 6)),
        }
        specs["labels" if labelled else "rowids"] = (
            ("uint8", (expected, 10)) if labelled else ("S64", (expected, 10))
        )
        out = {
            key: np.lib.format.open_memmap(folder / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
            for key, (dtype, shape) in specs.items()
        }
        # Padding must be initialized explicitly; don't rely on filesystem behavior.
        out["history"][:] = 0
        out["age_hours"][:] = 0
        seen_events = set()
        previous_ts = None
        clicks, empty, n = 0, 0, 0
        with csv_reader(source) as reader:
            expected_header = BASE_COLUMNS + (["clicked"] if labelled else [])
            if next(reader) != expected_header:
                raise ValueError(f"Unexpected columns in {source}")
            for eid, group in itertools.groupby(reader, key=lambda r: r[1]):
                rows = list(group)
                if n >= expected or len(rows) != 10 or eid in seen_events:
                    raise ValueError(f"Unexpected event count, size, or ordering: {eid}")
                seen_events.add(eid)
                if any(len(row) != len(expected_header) for row in rows):
                    raise ValueError(f"Wrong number of columns: {eid}")
                if any(row[2:3] + row[4:9] != rows[0][2:3] + rows[0][4:9] for row in rows):
                    raise ValueError(f"Inconsistent event context: {eid}")
                row = rows[0]
                ts = int(row[4])
                if labelled and previous_ts is not None and ts < previous_ts:
                    raise ValueError("Training events must be ordered by timestamp")
                previous_ts = ts
                hids, hts = json.loads(row[7]), json.loads(row[8])
                if len(hids) != len(hts) or len(hids) > 20:
                    raise ValueError(f"Invalid history lengths: {eid}")
                if any(t >= ts for t in hts) or any(a > b for a, b in zip(hts, hts[1:])):
                    raise ValueError(f"Invalid history timestamps: {eid}")
                out["users"][n] = encode(users, row[2])
                out["items"][n] = [encode(items, r[3]) for r in rows]
                out["timestamps"][n] = ts
                if len(set(out["items"][n].tolist())) != 10:
                    raise ValueError(f"Repeated candidate: {eid}")
                if hids:
                    out["history"][n, :len(hids)] = [encode(items, h) for h in hids]
                    out["age_hours"][n, :len(hids)] = [(ts - t) / 3600 for t in hts]
                else:
                    empty += 1
                hour = (ts % 86400) / 3600
                gap = (ts - hts[-1]) / 3600 if hts else 0
                span = (hts[-1] - hts[0]) / 3600 if hts else 0
                out["context"][n] = [
                    math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
                    float(row[6]), len(hids) / 20,
                    math.log1p(gap) / math.log1p(24 * 30),
                    math.log1p(span) / math.log1p(24 * 30),
                ]
                if labelled:
                    y = [int(r[9]) for r in rows]
                    if any(v not in (0, 1) for v in y) or sum(y) > 1:
                        raise ValueError(f"Invalid labels: {eid}")
                    out["labels"][n] = y
                    clicks += sum(y)
                else:
                    rowids = [r[0].encode("utf-8") for r in rows]
                    if any(len(r) > 64 for r in rowids):
                        raise ValueError("rowid exceeds cache storage width; increase S64")
                    out["rowids"][n] = rowids
                n += 1
                if n % 50_000 == 0:
                    print(f"prepare {name}: {n:,}/{expected:,} events", flush=True)
        if n != expected:
            raise ValueError(f"Expected {expected} events in {source}, found {n}")
        for array in out.values():
            array.flush()
        summaries[name] = {"events": n, "rows": n * 10, "empty_histories": empty}
        if labelled:
            summaries[name].update(clicks=clicks, ctr=clicks / (n * 10))
        print(f"prepared {name}: {summaries[name]}", flush=True)
    meta = {
        "version": 1, "cache_id": uuid.uuid4().hex, "sources": sources,
        "n_users": len(users) + 1, "n_items": len(items) + 1, "summary": summaries,
    }
    (cache / "vocabulary.json").write_text(json.dumps({"users": users, "items": items}), encoding="utf-8")
    # This is the completion marker: incomplete caches cannot be loaded.
    (cache / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Ready: {cache}; users={len(users):,}, items={len(items):,}", flush=True)
    return meta


def load_cache(cache, name):
    cache = Path(cache)
    if not (cache / "metadata.json").exists():
        raise ValueError("Run prepare first; this cache is missing or incomplete")
    meta = json.loads((cache / "metadata.json").read_text())
    keys = ARRAYS + (("labels",) if name == "train" else ("rowids",))
    data = {key: np.load(cache / name / f"{key}.npy", mmap_mode="r") for key in keys}
    return meta, data


def temporal_split(data, cutoff):
    """Fit strictly before cutoff; validate on each user's first event afterward."""
    fit = np.flatnonzero(data["timestamps"] < cutoff)
    later = np.flatnonzero(data["timestamps"] >= cutoff)
    if not len(fit) or not len(later):
        raise ValueError("Cutoff must leave nonempty training and validation periods")
    # prepare() checks chronological order; first occurrence is first event in time.
    _, first = np.unique(data["users"][later], return_index=True)
    valid = later[np.sort(first)]
    return fit, valid


def with_history_limit(data, limit=20):
    """Keep the most recent N clicks, updating all history-derived input features.

    This is the data experiment hook. It never uses labels and never changes the
    original cache. A limit of zero provides a clean no-history ablation.
    """
    if not 0 <= limit <= 20:
        raise ValueError("history_limit must be between 0 and 20")
    if limit == 20:
        return data
    result = dict(data)
    history = np.array(data["history"])
    ages = np.array(data["age_hours"])
    lengths = (history != 0).sum(1)
    keep = (history != 0) & (np.arange(20)[None, :] >= np.maximum(lengths-limit, 0)[:, None])
    history[~keep] = 0
    ages[~keep] = 0
    count = keep.sum(1)
    latest = np.where(keep, ages, np.inf).min(1)
    latest = np.where(count > 0, latest, 0)
    span = np.maximum(ages.max(1)-latest, 0)
    context = np.array(data["context"])
    context[:, 3] = count / 20
    context[:, 4] = np.log1p(latest) / np.log1p(24*30)
    context[:, 5] = np.log1p(span) / np.log1p(24*30)
    result.update(history=history, age_hours=ages, context=context)
    return result


def fit_priors(data, indices, n_users, n_items, smoothing=50):
    """All labels used here belong exclusively to the fitting period."""
    users = data["users"][indices]
    items = data["items"][indices]
    labels = data["labels"][indices].astype(np.float64)
    ctr = float(labels.mean())
    if not 0 < ctr < 1:
        raise ValueError("Training data must contain both classes")
    user_impr = np.bincount(users, minlength=n_users) * 10
    user_click = np.bincount(users, weights=labels.sum(1), minlength=n_users)
    item_impr = np.bincount(items.ravel(), minlength=n_items)
    item_click = np.bincount(items.ravel(), weights=labels.ravel(), minlength=n_items)
    logit = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1-1e-6))
    global_logit = float(logit(ctr))
    ub = logit((user_click + smoothing * ctr) / (user_impr + smoothing)) - global_logit
    ib = logit((item_click + smoothing * ctr) / (item_impr + smoothing)) - global_logit
    known_users = user_impr > 0
    known_items = item_impr > 0
    # History-only items also receive training gradients through history pooling.
    known_items[np.unique(data["history"][indices])] = True
    known_users[0] = known_items[0] = False
    ub[0] = ib[0] = 0
    return ctr, ub.astype("float32"), ib.astype("float32"), known_users, known_items
