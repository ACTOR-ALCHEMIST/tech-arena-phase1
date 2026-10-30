import unittest
import numpy as np
import torch
from torch import nn
from arena_native_attention import HistoryAttention


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(5))

    def forward(self, x):
        return torch.cat([x.sum(-1), x.mean((1, 2))[:, None]], 1)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        rng = np.random.default_rng(19)
        pre = dict(users=rng.normal(size=(9, 4)).astype('f'), items=rng.normal(size=(12, 4)).astype('f'), sizes=[2, 2])
        pre['users'][0] = pre['items'][0] = 0
        self.model = HistoryAttention(Base(), pre, np.zeros(7, 'f'), np.ones(7, 'f'),
                                      dict(width=32, fine_tune=True, history_layers=1, slate=True))
        self.x = torch.randn(3, 10, 7)
        self.batch = dict(users=torch.tensor([0, 1, 2]), items=torch.randint(0, 12, (3, 10)),
                          history=torch.tensor([[0]*20, [1, 2, 0, 3]+[0]*16, [5]*20]),
                          age_hours=torch.arange(20).float().expand(3, -1), context=torch.randn(3, 6))

    def test_zero_residual_and_cold_backward(self):
        self.model.eval()
        actual = self.model(self.batch, self.x)
        torch.testing.assert_close(actual, self.model.base(self.x[:, :, :5]))
        actual.square().mean().backward()
        for p in self.model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(torch.all(self.model.items.weight.grad[0] == 0))

    def test_candidate_equivariance_and_padding_invariance(self):
        with torch.no_grad():
            self.model.score[-1].weight.normal_(0, .03)
            self.model.outside[-1].weight.normal_(0, .03)
        self.model.eval()
        expected = self.model(self.batch, self.x)
        perm = torch.tensor([9, 0, 7, 1, 6, 2, 5, 3, 8, 4])
        changed = dict(self.batch, items=self.batch['items'][:, perm])
        actual = self.model(changed, self.x[:, perm])
        torch.testing.assert_close(actual[:, :10], expected[:, perm], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(actual[:, 10], expected[:, 10], atol=2e-6, rtol=1e-5)
        ages = self.batch['age_hours'].clone()
        ages[self.batch['history'] == 0] = 9999
        torch.testing.assert_close(self.model(dict(self.batch, age_hours=ages), self.x), expected)


if __name__ == '__main__':
    unittest.main()
