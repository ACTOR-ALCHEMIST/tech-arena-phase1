import unittest
import torch
from arena_objectives import loss


class RankingTests(unittest.TestCase):
    def test_ranking_gradient_improves_global_positive_negative_margin(self):
        torch.manual_seed(3)
        z = torch.randn(8, 11, requires_grad=True)
        y = torch.zeros(8, 10)
        y[0, 2] = y[3, 4] = y[6, 1] = 1
        ce = loss(z, y, dict(loss='event_ce'))
        ranked = loss(z, y, dict(loss='event_ce', rank_weight=1., rank_samples=256))
        grad, = torch.autograd.grad(ranked-ce, z)
        self.assertTrue(torch.isfinite(grad).all())

        def margin(value):
            s = value.softmax(1)[:, :10].logit()
            return s[y>0].mean()-s[y==0].mean()

        self.assertGreater(float(margin(z-.1*grad).detach()), float(margin(z).detach()))

    def test_no_positive_batch_keeps_finite_cross_entropy(self):
        z, y = torch.randn(3, 11), torch.zeros(3, 10)
        torch.testing.assert_close(loss(z, y, dict(loss='event_ce', rank_weight=1., rank_samples=256)),
                                   loss(z, y, dict(loss='event_ce')))

    def test_default_sampling_preserves_existing_recipe(self):
        z, y = torch.randn(5, 11), torch.zeros(5, 10)
        y[2, 1] = 1
        torch.manual_seed(84)
        old = loss(z, y, dict(loss='event_ce', rank_weight=.3))
        torch.manual_seed(84)
        explicit = loss(z, y, dict(loss='event_ce', rank_weight=.3, rank_samples=32))
        torch.testing.assert_close(old, explicit, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
