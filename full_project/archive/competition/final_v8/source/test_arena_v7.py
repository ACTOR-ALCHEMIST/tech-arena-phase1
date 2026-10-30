import unittest
import numpy as np
from arena_residual_adapter import initialize_graph_residual
from arena_windows import days


class V7Tests(unittest.TestCase):
    def test_graph_adapter_preserves_base_and_unknown_zero(self):
        rng = np.random.default_rng(3)
        base = {'items': rng.normal(size=(120, 8)).astype('f')}
        graph = {'items': rng.normal(size=(120, 32)).astype('f')}
        before = base['items'].copy()
        known = np.arange(120) > 5
        for rank in [8, 24]:
            x = initialize_graph_residual(base, graph, known, rank)
            self.assertEqual(x.shape, (120, rank))
            np.testing.assert_array_equal(x[~known], 0)
            np.testing.assert_allclose(np.linalg.norm(x[known], axis=1), .7, atol=1e-6)
        np.testing.assert_array_equal(base['items'], before)

    def test_expanding_origin_dates_follow_cutoff(self):
        config = {'origin_start': '2025-01-21'}
        self.assertEqual(days(config, '2025-01-24', 'origin'), 3)
        self.assertEqual(days(config, '2025-01-26', 'origin'), 5)
        self.assertEqual(days(config, '2025-01-27', 'origin'), 6)
        self.assertEqual(days({'origin_days': 3}, '2025-01-27', 'origin'), 3)
        with self.assertRaises(ValueError):
            days(config, '2025-01-21', 'origin')


if __name__ == '__main__':
    unittest.main()
