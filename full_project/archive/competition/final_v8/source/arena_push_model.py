"""Separate event-click propensity from conditional candidate choice."""
import copy
import numpy as np
import torch
from torch import nn
from arena_neural_v2 import LargeSlateModel, mlp


class GatedSlateModel(LargeSlateModel):
    def __init__(self, width, mean, scale, config):
        base = copy.deepcopy(config)
        base['architecture'] = 'v2_cross' if config['architecture'].endswith('cross') else 'v2_wide'
        assert base['loss'] == 'event_ce'
        super().__init__(width, mean, scale, base)
        self.gate_stats = mlp([5, 32, 1], config.get('dropout', .1))
        nn.init.zeros_(self.gate_stats[-1].weight)
        nn.init.zeros_(self.gate_stats[-1].bias)

    def forward(self, x):
        logits = super().forward(x)
        score, base_gate = logits[:, :10], logits[:, 10]
        # Gate learns from score summaries, without fitting candidate preference
        # vectors against no-click events whose relative ranking is unobserved.
        s = score.detach()
        top = s.topk(2, 1).values
        stats = torch.stack([s.logsumexp(1), s.mean(1), s.std(1, unbiased=False), top[:, 0], top[:, 0]-top[:, 1]], 1)
        gate = base_gate+self.gate_stats(stats).squeeze(1)
        outside = score.logsumexp(1)-gate
        return torch.cat([score, outside[:, None]], 1)


def build_model(width, mean, scale, config):
    if config.get('architecture', '').startswith('v2_gate_'):
        return GatedSlateModel(width, mean, scale, config)
    from arena_tabular import build_model as original
    return original(width, mean, scale, config)


def normalizer(values, config):
    if config.get('normalizer') != 'relative_range':
        from arena_tabular import normalizer as original
        return original(values, config.get('constant_policy', 'zero'))
    total = np.zeros(values.shape[1], np.float64)
    squared = total.copy()
    low, high = np.full(values.shape[1], np.inf), np.full(values.shape[1], -np.inf)
    for start in range(0, len(values), 100000):
        b = np.asarray(values[start:start+100000])
        total += b.sum(0, dtype=np.float64)
        squared += np.einsum('ij,ij->j', b, b, dtype=np.float64)
        low, high = np.minimum(low, b.min(0)), np.maximum(high, b.max(0))
    mean = total/len(values)
    variance = np.maximum(squared/len(values)-mean*mean, 1e-24)
    constant = (high-low) <= 1e-6*np.maximum(np.maximum(np.abs(high), np.abs(low)), 1e-12)
    scale = np.where(constant, 1e12, np.sqrt(variance))
    return mean.astype('f'), scale.astype('f')
