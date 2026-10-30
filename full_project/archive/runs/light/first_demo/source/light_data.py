"""Small, reproducible user cohorts for quick data/model experiments."""

import hashlib

import numpy as np

from ctr_data import load_cache, temporal_split


def build_light_dataset(cache, cutoff, cohort_users=4000, split_seed=2026):
    """Select users without looking at labels; retain their complete fitting period.

    The subset is intentionally a user-cohort experiment. Its AUC should be
    compared with other runs on the same split_id, not with full-data scores.
    Sampling depends on split_seed, independently of the model's training seed.
    """
    if cohort_users < 1:
        raise ValueError("cohort_users must be positive")
    meta, full = load_cache(cache, "train")
    fit, valid = temporal_split(full, cutoff)
    rng = np.random.default_rng(split_seed)
    source_valid = np.sort(rng.choice(valid, min(cohort_users, len(valid)), replace=False))
    selected_users = full["users"][source_valid]
    source_fit = fit[np.isin(full["users"][fit], selected_users)]
    if not len(source_fit):
        raise ValueError("The selected cohort has no fitting events; increase cohort_users")
    source_indices = np.concatenate([source_fit, source_valid])
    # Materialize only this cohort; the full arrays are memory-mapped.
    data = {key: np.array(array[source_indices]) for key, array in full.items()}
    local_fit = np.arange(len(source_fit))
    local_valid = np.arange(len(source_fit), len(source_indices))
    digest = hashlib.sha256(meta["cache_id"].encode())
    for indices in (source_fit, source_valid):
        digest.update(np.asarray(indices, dtype="<i8").tobytes())
    summary = {
        "split_id": digest.hexdigest(), "cache_id": meta["cache_id"],
        "split_seed": split_seed, "cutoff_utc_timestamp": cutoff,
        "cohort_users": len(selected_users), "fit_events": len(source_fit),
        "validation_events": len(source_valid),
        "data_mb": sum(a.nbytes for a in data.values()) / 1e6,
    }
    return data, (local_fit, local_valid), source_indices, summary
