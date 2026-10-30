import unittest
import numpy as np
import torch
from arena_spherical_probe import Spherical


class SphericalTests(unittest.TestCase):
    def test_known_mask_and_finite_cold_gradients(self):
        torch.manual_seed(93)
        init = dict(users=np.ones((4, 3), 'float32'), items=np.ones((8, 3), 'float32'), bias=np.zeros(8, 'float32'))
        net = Spherical(init, np.array([0, 1, 1, 0], bool), np.array([0, 1, 1, 1, 0, 0, 0, 0], bool))
        loss = net.loss(torch.tensor([1, 2]), torch.tensor([1, 2]), torch.tensor([[0, 4, 7], [2, 3, 5]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
        rep = net.representation()
        np.testing.assert_array_equal(rep['items'][[0, 4, 5, 6, 7]], 0)
        np.testing.assert_array_equal(rep['users'][[0, 3]], 0)
        np.testing.assert_allclose(np.linalg.norm(rep['items'][1:4], axis=1), 1., atol=1e-6)


if __name__ == '__main__':
    unittest.main()
