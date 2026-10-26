"""The two inference-only networks used by the frozen V31 feature/head recipe."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(widths, dropout=.1, final_activation=False):
    layers = []
    for i, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(left, right))
        if i < len(widths)-2 or final_activation:
            layers.extend([nn.SiLU(), nn.Dropout(dropout)])
    return nn.Sequential(*layers)


class WideSlate(nn.Module):
    """323 numeric features -> slate context -> 10 item logits + no-click logit."""
    def __init__(self, saved):
        super().__init__()
        config, state, width = saved['config'], saved['state_dict'], saved['width']
        if config['architecture'] != 'v2_wide' or width != 323:
            raise ValueError('This package only supports the frozen V31 wide head')
        self.register_buffer('mean', state['mean'].clone())
        self.register_buffer('scale', state['scale'].clone())
        dropout, hidden, context_width = config['dropout'], config['hidden'], config['context_width']
        self.context = mlp([3*width, 128, context_width], dropout, True)
        self.deep = mlp([width+context_width]+hidden, dropout, True)
        self.score = nn.Linear(hidden[-1], 1)
        self.outside = mlp([3*width, 128, 64, 1], dropout)
        self.load_state_dict(state, strict=True)
        self.eval()

    def forward(self, x):
        x = ((x-self.mean)/self.scale).clamp(-6, 6)
        aggregate = torch.cat([x.mean(1), x.max(1).values, x.std(1, unbiased=False)], 1)
        values = torch.cat([x, self.context(aggregate)[:, None].expand(-1, x.shape[1], -1)], 2)
        scores = self.score(self.deep(values)).squeeze(2)
        return torch.cat([scores, self.outside(aggregate)], 1)

    @torch.inference_mode()
    def predict(self, x):
        parts = []
        for offset in range(0, len(x)//10, 2048):
            batch = np.array(x[offset*10:(offset+2048)*10]).reshape(-1, 10, x.shape[1])
            parts.append(self(torch.tensor(batch)).softmax(1)[:, :10].numpy())
        return np.concatenate(parts)


class FreshMF(nn.Module):
    """Fresh exposure-trained MF, used only to construct V31 numeric features."""
    def __init__(self, state):
        super().__init__()
        names = {'users.weight': 'users', 'items.weight': 'items',
                 'user_delta.weight': 'user_delta', 'item_delta.weight': 'item_delta'}
        for key, value in state.items():
            self.register_buffer(names.get(key, key), value)
        self.eval()

    def representation(self):
        return dict(users=self.users.numpy(), items=self.items.numpy(),
                    bias=self.ib.numpy()*self.coefficients.numpy()[2]+self.item_delta.numpy().ravel())

    def user_offset(self):
        return self.ub.numpy()*self.coefficients.numpy()[1]+self.user_delta.numpy().ravel()

    def forward(self, batch):
        uid, ids, hist = batch['users'], batch['items'], batch['history']
        age, ctx = batch['age_hours'], batch['context']
        dot = (self.users[uid, None]*self.items[ids]).sum(2)
        v, h = F.normalize(self.v0[ids], dim=2, eps=1e-6), F.normalize(self.v0[hist], dim=2, eps=1e-6)
        similarity = torch.bmm(v, h.transpose(1, 2))
        visible, length = hist != 0, (hist != 0).sum(1)
        mean = (similarity*visible[:, None]).sum(2)/length.clamp_min(1)[:, None]
        peak = similarity.masked_fill(~visible[:, None], -1).max(2).values
        peak = torch.where(length[:, None] > 0, peak, 0)
        weight = torch.exp(-age/72)*visible
        recent = (similarity*weight[:, None]).sum(2)/weight.sum(1).clamp_min(1e-6)[:, None]
        repeat = (ids[:, :, None] == hist[:, None]) & visible[:, None]
        values = [dot, self.ub[uid, None].expand_as(dot), self.ib[ids], mean, peak, recent,
                  repeat.any(2).float(), repeat.sum(2).float().log1p(),
                  (repeat*torch.exp(-age[:, None]/12)).sum(2), (repeat*torch.exp(-age[:, None]/72)).sum(2)]
        features = torch.cat([torch.stack(values, 2), ctx[:, None].expand(-1, 10, -1)], 2)
        score = (features*self.coefficients).sum(2)
        score = score+self.user_delta[uid]+self.item_delta[ids].squeeze(2)
        return torch.cat([score, self.outside.expand(len(score), 1)], 1)

    @torch.inference_mode()
    def predict(self, data):
        parts = []
        for start in range(0, len(data['users']), 2048):
            batch = {k: torch.as_tensor(np.array(data[k][start:start+2048]),
                                       dtype=torch.long if k in ['users', 'items', 'history'] else torch.float32)
                     for k in ['users', 'items', 'history', 'age_hours', 'context']}
            parts.append(self(batch).softmax(1)[:, :10].numpy())
        return np.concatenate(parts)
