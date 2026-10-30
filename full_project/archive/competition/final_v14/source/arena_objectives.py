"""Event-preserving losses for user-balanced training and global AUC ranking."""
import numpy as np
import torch
from torch.nn import functional as F


def user_weights(users, power):
    if not power:
        return None
    counts = np.bincount(users)
    w = counts[users].astype(np.float64)**(-power)
    w /= w.mean()
    w = np.clip(w, .2, 3.)
    return (w/w.mean()).astype(np.float32)


def loss(logits, labels, config, weights=None):
    if config['loss'] == 'event_ce':
        target = torch.where(labels.sum(1) > 0, labels.argmax(1), 10)
        if weights is None:
            value = F.cross_entropy(logits, target)
        else:
            value = (F.cross_entropy(logits, target, reduction='none')*weights).mean()
    else:
        value = F.binary_cross_entropy_with_logits(logits, labels)
    if config.get('rank_weight', 0):
        # Rank clicked vs non-clicked rows across complete events in this batch.
        # log-odds has the same ordering as the submitted softmax probability.
        p = logits.softmax(1)[:, :10].clamp(1e-6, 1-1e-6)
        s = (p.log()-torch.log1p(-p)).flatten()
        flat = labels.flatten() > 0
        pos, neg = s[flat], s[~flat]
        if len(pos) and len(neg):
            take = torch.randint(len(neg), (len(pos), 32), device=logits.device)
            ranking = F.softplus(neg[take]-pos[:, None]).mean()
            value = value+config['rank_weight']*ranking
    return value
