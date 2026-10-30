import unittest
import numpy as np
import torch
from torch.nn import functional as F
from arena_laplace_solver import terms, solve_user, solve_users


class LaplaceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(191)
        self.design = rng.normal(size=(21, 10, 3))
        self.design[:, :, -1] = 1
        self.offsets = rng.normal(size=(21, 11))
        self.target = rng.integers(0, 11, 21)
        self.precision = np.array([3., 2., 1.])

    def test_gradient_and_hessian_match_finite_differences(self):
        delta = np.array([.2, -.3, .4])
        value, gradient, hessian = terms(delta, self.design, self.offsets, self.target, self.precision)
        eps = 1e-5
        for k in range(3):
            unit = np.zeros(3)
            unit[k] = eps
            plus = terms(delta+unit, self.design, self.offsets, self.target, self.precision)
            minus = terms(delta-unit, self.design, self.offsets, self.target, self.precision)
            self.assertAlmostEqual((plus[0]-minus[0])/(2*eps), gradient[k], places=7)
            np.testing.assert_allclose((plus[1]-minus[1])/(2*eps), hessian[:, k], atol=1e-7)
        self.assertTrue(np.linalg.eigvalsh(hessian).min() > 0)

    def test_map_matches_independent_torch_optimizer(self):
        delta, cov, value, gradient, steps = solve_user(self.design, self.offsets, self.target, self.precision)
        d, offset = torch.tensor(self.design), torch.tensor(self.offsets)
        prior, target = torch.tensor(self.precision), torch.tensor(self.target)
        parameter = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([parameter], max_iter=100, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            scores = torch.cat([offset[:, :10]+torch.einsum('bik,k->bi', d, parameter), offset[:, 10:]], 1)
            loss = F.cross_entropy(scores, target, reduction='sum')+.5*(prior*parameter.square()).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        np.testing.assert_allclose(delta, parameter.detach().numpy(), atol=1e-6)
        self.assertLess(gradient, 1e-6)
        self.assertAlmostEqual(value, float(closure().detach()), places=8)
        np.testing.assert_allclose(cov@terms(delta, self.design, self.offsets, self.target, self.precision)[2], np.eye(3), atol=1e-12)

    def test_no_observations_returns_prior_without_invented_signal(self):
        delta, cov, value, gradient, steps = solve_user(np.empty((0, 10, 3)), np.empty((0, 11)), np.empty(0, np.int64), self.precision)
        np.testing.assert_array_equal(delta, 0)
        np.testing.assert_allclose(cov, np.diag(1/self.precision), atol=1e-14)
        self.assertEqual(value, 0)

    def test_grouped_solver_matches_individual_users(self):
        rng = np.random.default_rng(81)
        vectors = rng.normal(size=(12, 2))
        items = rng.integers(0, 12, (6, 10))
        offsets = rng.normal(size=(6, 11))
        target = rng.integers(0, 11, 6)
        ptr = np.array([0, 2, 2, 6])
        precision = np.tile(self.precision, (3, 1))
        d, c, *_ = solve_users(ptr, items, offsets, target, vectors, 1.7, precision)
        for user in range(3):
            a, b = ptr[user:user+2]
            design = np.concatenate([1.7*vectors[items[a:b]], np.ones((b-a, 10, 1))], 2)
            expected = solve_user(design, offsets[a:b], target[a:b], precision[user])
            np.testing.assert_allclose(d[user], expected[0], atol=1e-7)
            np.testing.assert_allclose(c[user], expected[1], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
