import unittest
import numpy as np
import torch
from arena_daily import EventSubset, StackRows
from arena_exposure_repr import ExposureFactors


class Push084Tests(unittest.TestCase):
    def test_small_graph_scores_are_variable_not_constant(self):
        from arena_push_model import normalizer
        x = np.array([[1e-6, 0., 1.], [2e-6, 0., 1.], [3e-6, 0., 1.]], np.float32)
        mean, scale = normalizer(x, dict(normalizer='relative_range'))
        self.assertLess(scale[0], 1e-5)
        self.assertEqual(scale[1], np.float32(1e12))
        self.assertEqual(scale[2], np.float32(1e12))
        z = (x-mean)/scale
        self.assertAlmostEqual(float(z[:, 0].std()), 1., places=5)
        np.testing.assert_array_equal(z[:, 1:], 0)

    def test_conditional_partition_matches_full_catalog(self):
        from arena_softmax_repr import partition_loss
        torch.manual_seed(4)
        u, v = torch.randn(3, 8, requires_grad=True), torch.randn(5, 8, requires_grad=True)
        b = torch.randn(5, requires_grad=True)
        target = torch.tensor([1, 3, 4])
        expected = torch.nn.functional.cross_entropy(u@v.T+b[None], target)
        actual = partition_loss(u, v[target], b[target], v, b, 6, torch.arange(5)[None] == target[:, None])
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(u.grad).all())
        self.assertTrue(torch.isfinite(v.grad).all())

    def test_no_click_gate_does_not_train_candidate_preference(self):
        from arena_push_model import build_model
        cfg = dict(architecture='v2_gate_cross', loss='event_ce', dropout=0., hidden=[32, 16], context_width=8, cross_layers=2)
        m = build_model(7, np.zeros(7, 'f'), np.ones(7, 'f'), cfg)
        x = torch.randn(4, 10, 7)
        z = m(x)
        perm = torch.randperm(10)
        shuffled = m(x[:, perm])
        torch.testing.assert_close(z[:, :10][:, perm], shuffled[:, :10], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(z[:, 10], shuffled[:, 10], atol=1e-6, rtol=1e-5)
        torch.nn.functional.cross_entropy(z, torch.full((4,), 10)).backward()
        self.assertLess(m.score.weight.grad.abs().max().item(), 1e-6)

    def test_row_composition_retains_labels_and_order(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=(100, 7)).astype('f'), rng.normal(size=(30, 7)).astype('f')
        p = EventSubset(a, np.array([3, 7, 1]))
        expected = np.concatenate([a.reshape(10, 10, 7)[[3, 7, 1]].reshape(-1, 7), b])
        x = StackRows(p, b)
        order = rng.permutation(len(x))
        np.testing.assert_array_equal(x[order], expected[order])
        np.testing.assert_array_equal(x[18:43], expected[18:43])

    def test_exposure_empty_history_gradients_and_permutation(self):
        rng = np.random.default_rng(9)
        u, v = rng.normal(size=(4, 8)).astype('f'), rng.normal(size=(14, 8)).astype('f')
        u[0], v[0] = 0, 0
        m = ExposureFactors(dict(users=u, items=v, bias=np.zeros(14, 'f')),
                            dict(pop=np.r_[0, np.ones(13)].astype('f'), ub=np.zeros(4, 'f'), ib=np.zeros(14, 'f')))
        uid = torch.tensor([0, 1])
        ids = torch.arange(1, 11)[None].expand(2, -1)
        h, age, ctx = torch.zeros(2, 20, dtype=torch.long), torch.zeros(2, 20), torch.zeros(2, 6)
        z = m(uid, ids, h, age, ctx)
        self.assertTrue(torch.isfinite(z).all())
        perm = torch.randperm(10)
        shuffled = m(uid, ids[:, perm], h, age, ctx)
        torch.testing.assert_close(z[:, :10][:, perm], shuffled[:, :10])
        torch.testing.assert_close(z[:, 10], shuffled[:, 10])
        loss = torch.nn.functional.cross_entropy(z, torch.tensor([10, 3]))
        loss.backward()
        for p in m.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        torch.testing.assert_close(m.items.weight.grad[0], torch.zeros(8))


if __name__ == '__main__':
    unittest.main()
