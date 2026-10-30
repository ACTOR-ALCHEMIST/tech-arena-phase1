import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from arena_ood import corrupt


class HistoryAugmentationTest(unittest.TestCase):
    def data(self):
        h = np.zeros((3, 20), np.int32)
        a = np.zeros((3, 20), np.float32)
        h[0, :5] = [1, 2, 3, 4, 5]
        a[0, :5] = [50, 40, 30, 20, 10]
        h[1, :2] = [6, 7]
        a[1, :2] = [8, 2]
        return dict(history=h, age_hours=a, context=np.zeros((3, 6), np.float32),
                    users=np.arange(1, 4), items=np.arange(30).reshape(3, 10))

    def test_delete_alignment_compaction_and_no_mutation(self):
        data = self.data()
        original = {k: v.copy() for k, v in data.items()}
        result = corrupt(data, {'mode': 'thin', 'drop_probability': .5}, 18)
        again = corrupt(data, {'mode': 'thin', 'drop_probability': .5}, 18)
        for k in data:
            np.testing.assert_array_equal(data[k], original[k])
            np.testing.assert_array_equal(result[k], again[k])
        for h, ages, before_h, before_a in zip(result['history'], result['age_hours'], data['history'], data['age_hours']):
            length = np.count_nonzero(h)
            self.assertTrue((h[:length] != 0).all())
            self.assertTrue((h[length:] == 0).all())
            pairs = dict(zip(before_h, before_a))
            for item, age in zip(h[:length], ages[:length]):
                self.assertEqual(age, pairs[item])
        np.testing.assert_allclose(result['context'][:, 3], np.count_nonzero(result['history'], axis=1)/20, rtol=1e-6)

    def test_recent_history_context_and_empty_history(self):
        original = self.data()
        unchanged = corrupt(original, {'mode': 'thin', 'drop_probability': 0})
        for key in original:
            np.testing.assert_array_equal(unchanged[key], original[key])
        result = corrupt(self.data(), {'mode': 'truncate', 'limits': [2]})
        np.testing.assert_array_equal(result['history'][0, :3], [4, 5, 0])
        self.assertAlmostEqual(float(result['context'][0, 4]), np.log1p(10)/np.log1p(720), places=6)
        self.assertAlmostEqual(float(result['context'][0, 5]), np.log1p(10)/np.log1p(720), places=6)
        empty = corrupt(self.data(), {'mode': 'thin', 'drop_probability': 1})
        self.assertFalse(empty['history'].any())
        self.assertFalse(empty['age_hours'].any())
        self.assertFalse(empty['context'][:, 3:].any())
        self.assertTrue(np.isfinite(empty['context']).all())

    def test_cold_user_retains_all_candidates_and_observed_history(self):
        data = self.data()
        result = corrupt(data, {'mode': 'cold_user'})
        self.assertFalse(result['users'].any())
        np.testing.assert_array_equal(result['items'], data['items'])
        np.testing.assert_array_equal(result['history'], data['history'])

    def test_new_features_rotation_and_candidate_permutation(self):
        from arena_ood_features import extra_features
        data = self.data()
        data['timestamps'] = np.full(3, 1737763200, dtype=np.int64)
        rng = np.random.default_rng(714)
        factors = {'users': rng.normal(size=(4, 8)).astype(np.float32),
                   'items': rng.normal(size=(30, 8)).astype(np.float32)}
        factors['users'][0] = 0
        factors['items'][0] = 0
        config = {'sgns_factors': [8]}
        with patch('arena_sgns_features.embedding_model', return_value=factors):
            x, names = extra_features(data, np.arange(3), '2025-01-24', config)
            permutation = rng.permutation(10)
            permuted = dict(data, items=data['items'][:, permutation])
            xp, _ = extra_features(permuted, np.arange(3), '2025-01-24', config)
        np.testing.assert_allclose(x.reshape(3, 10, -1)[:, permutation], xp.reshape(3, 10, -1), atol=2e-6)
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        rotated = {k: (v@rotation).astype(np.float32) for k, v in factors.items()}
        with patch('arena_sgns_features.embedding_model', return_value=rotated):
            xr, _ = extra_features(data, np.arange(3), '2025-01-24', config)
        np.testing.assert_allclose(x, xr, atol=2e-6)
        self.assertEqual(len(names), 20)
        self.assertTrue(np.isfinite(x).all())

    def test_consensus_is_invariant_to_independent_model_coordinates(self):
        from arena_ood_consensus import extra_features
        data = self.data()
        data['timestamps'] = np.full(3, 1737763200, dtype=np.int64)
        rng = np.random.default_rng(9287)
        a = {'users': rng.normal(size=(4, 8)).astype(np.float32), 'items': rng.normal(size=(30, 8)).astype(np.float32)}
        b = {k: v+rng.normal(scale=.2, size=v.shape).astype(np.float32) for k, v in a.items()}
        for model in [a, b]:
            model['users'][0] = 0
            model['items'][0] = 0
        config = {'sgns_epochs': {'8': 1}, 'sgns_recipes': {'8': {'seed': 1}},
                  'sgns_extra': [{'factors': 8, 'recipe': {'seed': 2}}]}
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        rotated = {k: (v@rotation).astype(np.float32) for k, v in b.items()}
        with TemporaryDirectory() as temp:
            np.savez(Path(temp)/'stats.npz', pop=np.r_[0, np.ones(29)], known_users=np.array([False, True, True, True]))
            with patch('arena_core.snapshot', return_value=Path(temp)):
                with patch('arena_sgns_features.embedding_model', side_effect=[a, b]):
                    x, names = extra_features(data, np.arange(3), '2025-01-24', config)
                with patch('arena_sgns_features.embedding_model', side_effect=[a, rotated]):
                    xr, _ = extra_features(data, np.arange(3), '2025-01-24', config)
        self.assertEqual(len(names), 20)
        np.testing.assert_allclose(x, xr, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
