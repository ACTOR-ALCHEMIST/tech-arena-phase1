"""A small history-aware two-tower CTR model, plus explicit repeat features."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class HistoryTwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim=64, user_dropout=0.1,
                 use_history_pooling=True, use_repeat_features=True):
        super().__init__()
        self.user_dropout = user_dropout
        self.use_history_pooling = use_history_pooling
        self.use_repeat_features = use_repeat_features
        self.user_embedding = nn.Embedding(n_users, dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.user_bias = nn.Embedding(n_users, 1, padding_idx=0)
        self.item_bias = nn.Embedding(n_items, 1, padding_idx=0)
        self.user_network = nn.Sequential(nn.Linear(dim * 2 + 6, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.context_bias = nn.Linear(6, 1)
        self.repeat_head = nn.Linear(3, 1, bias=False)
        self.global_bias = nn.Parameter(torch.tensor(-2.7))
        # softplus(-1.35) is approximately log(2)/3: initial half-life ~3 days.
        self.raw_decay = nn.Parameter(torch.tensor(-1.35))
        self.register_buffer("known_users", torch.zeros(n_users, dtype=torch.bool))
        self.register_buffer("known_items", torch.zeros(n_items, dtype=torch.bool))
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        for layer in (self.user_bias, self.item_bias):
            nn.init.zeros_(layer.weight)
        for layer in (self.user_network[-1], self.context_bias, self.repeat_head):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    def initialize_priors(self, ctr, user_bias, item_bias, known_users, known_items):
        with torch.no_grad():
            self.global_bias.fill_(math.log(ctr / (1-ctr)))
            self.user_bias.weight[:, 0].copy_(torch.as_tensor(user_bias))
            self.item_bias.weight[:, 0].copy_(torch.as_tensor(item_bias))
            self.known_users.copy_(torch.as_tensor(known_users))
            self.known_items.copy_(torch.as_tensor(known_items))

    def forward(self, batch):
        uid = batch["users"]
        uid = torch.where(self.known_users[uid], uid, 0)
        if self.training and self.user_dropout:
            uid = torch.where(torch.rand_like(uid, dtype=torch.float32) < self.user_dropout, 0, uid)
        raw_items, raw_history = batch["items"], batch["history"]
        iid = torch.where(self.known_items[raw_items], raw_items, 0)
        history = torch.where(self.known_items[raw_history], raw_history, 0)
        weights = torch.exp(-F.softplus(self.raw_decay) * batch["age_hours"] / 24)
        weights = weights * (history != 0)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        pooled = (self.item_embedding(history) * weights.unsqueeze(-1)).sum(1)
        if not self.use_history_pooling:
            pooled = torch.zeros_like(pooled)
        user = self.user_embedding(uid)
        user = user + pooled + self.user_network(torch.cat([user, pooled, batch["context"]], 1))
        score = (user.unsqueeze(1) * self.item_embedding(iid)).sum(-1)
        score = score + self.global_bias + self.user_bias(uid) + self.item_bias(iid).squeeze(-1)
        score = score + self.context_bias(batch["context"])
        if not self.use_repeat_features:
            return score
        # These are observed context features, independent of the current label.
        match = (raw_items.unsqueeze(-1) == raw_history.unsqueeze(1)) & (raw_history.unsqueeze(1) != 0)
        count = match.sum(-1).float()
        recent = (match.float() / (1 + batch["age_hours"].unsqueeze(1) / 24)).sum(-1)
        repeat = torch.stack([(count > 0).float(), torch.log1p(count), recent], -1)
        return score + self.repeat_head(repeat).squeeze(-1)
