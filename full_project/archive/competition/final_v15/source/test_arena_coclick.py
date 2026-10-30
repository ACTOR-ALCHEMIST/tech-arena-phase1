import unittest
from unittest.mock import patch
import numpy as np
from arena_coclick import pairs, extra_features


class CoClickTest(unittest.TestCase):
    def test_pairs_sort_time_and_never_cross_users(self):
        clicks = {'users': np.array([2, 1, 1, 2, 1]), 'items': np.array([7, 4, 3, 8, 4]),
                  'minutes': np.array([5, 3, 1, 2, 2])}
        a, b = pairs(clicks, 1)
        self.assertEqual(list(zip(a, b)), [(3, 4), (8, 7)])

    def test_empty_history_and_permutation(self):
        rng = np.random.default_rng(1)
        rep = {'items': rng.normal(size=(16, 4)).astype('f'), 'users': rng.normal(size=(3, 4)).astype('f'),
               'bias': rng.normal(size=16).astype('f')}
        rep['items'][0] = 0
        data = dict(items=np.array([np.arange(1, 11), np.arange(2, 12)]), users=np.array([1, 2]),
                    history=np.zeros((2, 20), dtype='i'), age_hours=np.zeros((2, 20), dtype='f'))
        data['history'][1, :3] = [3, 4, 5]
        with patch('arena_coclick.model', return_value=rep):
            x, names = extra_features(data, np.arange(2), '2025-01-24')
            perm = rng.permutation(10)
            shuffled = dict(data, items=data['items'][:, perm])
            y, columns = extra_features(shuffled, np.arange(2), '2025-01-24')
        self.assertEqual(names, columns)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(x.reshape(2, 10, -1)[:, perm], y.reshape(2, 10, -1), atol=1e-6)
        for col, name in enumerate(names):
            if 'hist_' in name or 'last' in name or 'decay' in name:
                np.testing.assert_array_equal(x[:10, col], 0)


if __name__ == '__main__':
    unittest.main()
