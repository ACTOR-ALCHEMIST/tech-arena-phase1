"""Larger event scorers; no candidate position or identifier features.

The cross branch follows the matrix cross equation in DCN V2
(https://arxiv.org/abs/2008.13535). Attention operates over the ten
candidates, with no positional encoding, and is permutation equivariant.
"""
import numpy as np
import torch
from torch import nn


def mlp(widths, dropout=0.1, final_activation=False):
    layers = []
    for i, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(left, right))
        if i < len(widths) - 2 or final_activation:
            layers.extend([nn.SiLU(), nn.Dropout(dropout)])
    return nn.Sequential(*layers)


class LargeSlateModel(nn.Module):
    def __init__(self, width, mean, scale, config):
        super().__init__()
        self.loss = config.get('loss', 'event_ce')
        self.architecture = config['architecture']
        self.register_buffer('mean', torch.tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(np.asarray(scale), dtype=torch.float32))
        dropout = config.get('dropout', .1)
        hidden = config.get('hidden', [256, 128, 64])
        context_width = config.get('context_width', 0)
        self.context = mlp([3*width, 128, context_width], dropout, True) if context_width else None
        self.deep = mlp([width+context_width] + hidden, dropout, True)
        self.cross = nn.ModuleList()
        if self.architecture == 'v2_cross':
            self.cross = nn.ModuleList([nn.Linear(width, width) for _ in range(config.get('cross_layers', 2))])
            # Begin close to identity instead of multiplying large random crosses.
            for layer in self.cross:
                nn.init.normal_(layer.weight, std=.01)
                nn.init.zeros_(layer.bias)
        attention_width = config.get('attention_width', 128)
        self.attention = None
        if self.architecture == 'v2_attention':
            self.project = nn.Linear(width, attention_width)
            layer = nn.TransformerEncoderLayer(
                attention_width, config.get('heads', 4), attention_width*2,
                dropout, activation='gelu', batch_first=True, norm_first=True)
            self.attention = nn.TransformerEncoder(layer, config.get('attention_layers', 2),
                                                   enable_nested_tensor=False)
        output_width = hidden[-1] + (width if self.cross else 0) + (attention_width if self.attention is not None else 0)
        self.score = nn.Linear(output_width, 1)
        self.outside = mlp([3*width, 128, 64, 1], dropout)

    def forward(self, x):
        x = ((x-self.mean)/self.scale).clamp(-6, 6)
        aggregate = torch.cat([x.mean(1), x.max(1).values, x.std(1, unbiased=False)], 1)
        values = x
        if self.context is not None:
            values = torch.cat([x, self.context(aggregate)[:, None].expand(-1, x.shape[1], -1)], 2)
        branches = [self.deep(values)]
        if self.cross:
            crossed = x
            for layer in self.cross:
                crossed = x*layer(crossed) + crossed
            branches.append(crossed)
        if self.attention is not None:
            branches.append(self.attention(self.project(x)))
        scores = self.score(torch.cat(branches, 2)).squeeze(2)
        return torch.cat([scores, self.outside(aggregate)], 1) if self.loss == 'event_ce' else scores

    def probabilities(self, x):
        logits = self(x)
        return logits.softmax(1)[:, :10] if self.loss == 'event_ce' else logits.sigmoid()
