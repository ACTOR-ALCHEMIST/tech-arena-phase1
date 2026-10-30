import unittest
import numpy as np
from scipy.special import expit
from arena_online_events import build, features


class OnlineTests(unittest.TestCase):
    def setUp(self):
        self.t = 1737331200
        # Different users share item1; source timestamps include ties.
        self.raw = dict(users=np.array([1, 1, 2, 1, 2, 1], 'int32'),
                        items=np.tile(np.arange(1, 11), (6, 1)).astype('int32'),
                        timestamps=self.t+np.array([0, 60, 60, 120, 180, 240]),
                        labels=np.zeros((6, 10), 'uint8'))
        self.raw['labels'][[0, 2, 4], 0] = 1

    def get(self, raw=None, cap=None, queries=None):
        raw = raw or self.raw
        cap = self.t+300 if cap is None else cap
        ix = np.arange(6) if queries is None else queries
        index, meta = build(raw, cap, 4, 12)
        return features(index, raw, ix, cap), meta

    def test_counts_and_strict_ties_match_brute_force(self):
        (x, names), meta = self.get()
        z = x.reshape(6, 10, -1)
        for e in range(6):
            past = self.raw['timestamps'] < self.raw['timestamps'][e]
            same = past & (self.raw['users'] == self.raw['users'][e])
            prior = (self.raw['labels'][past].sum()+6)/(past.sum()+10)
            uc = self.raw['labels'][same].sum()
            self.assertAlmostEqual(np.expm1(z[e, 0, 1]), same.sum(), places=5)
            self.assertAlmostEqual(np.expm1(z[e, 0, 2]), uc, places=5)
            self.assertAlmostEqual(expit(z[e, 0, 3]), (uc+5*prior)/(same.sum()+5), places=6)
            item_start = names.index('online_item_known')
            self.assertAlmostEqual(np.expm1(z[e, 0, item_start+1]), past.sum(), places=5)
            self.assertAlmostEqual(np.expm1(z[e, 0, item_start+2]), self.raw['labels'][past, 0].sum(), places=5)
        self.assertLess(meta['max_fitting_timestamp'], self.t+300)

    def test_current_and_future_label_mutation_has_no_effect(self):
        (expected, _), _ = self.get(queries=np.array([1]))
        changed = dict(self.raw, labels=self.raw['labels'].copy())
        changed['labels'][changed['timestamps'] >= self.t+60] = 0
        changed['labels'][1:, 3] = 1
        (actual, _), _ = self.get(changed, queries=np.array([1]))
        np.testing.assert_array_equal(expected, actual)

    def test_held_out_other_users_item_and_global_labels_excluded(self):
        cap = self.t+120
        (expected, _), meta = self.get(cap=cap, queries=np.array([5]))
        changed = dict(self.raw, labels=self.raw['labels'].copy())
        changed['labels'][3:] = 0
        changed['labels'][3:, 9] = 1
        (actual, _), _ = self.get(changed, cap=cap, queries=np.array([5]))
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual(meta['fitting_events'], 3)

    def test_tie_source_permutation_is_invariant(self):
        # Put same user in both tied events so its streak is also tested.
        raw = dict(self.raw, users=self.raw['users'].copy())
        raw['users'][2] = 1
        index, _ = build(raw, self.t+300, 4, 12)
        expected, _ = features(index, raw, np.arange(6), self.t+300)
        permutation = np.array([5, 2, 0, 4, 1, 3])
        other = {k:v[permutation] for k, v in raw.items()}
        rebuilt, _ = build(other, self.t+300, 4, 12)
        actual, _ = features(rebuilt, raw, np.arange(6), self.t+300)
        np.testing.assert_array_equal(expected, actual)

    def test_cold_and_empty_past_without_query_targets(self):
        index, _ = build(self.raw, self.t, 4, 12)
        query = dict(users=np.array([0, 3], 'int32'), items=np.zeros((2, 10), 'int32'),
                     timestamps=np.array([self.t, self.t+100]))
        x, names = features(index, query, np.arange(2), self.t)
        self.assertEqual(x.shape, (20, 43))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(len(set(names)), 43)
        np.testing.assert_array_equal(x[:, 0], 0)


if __name__ == '__main__':
    unittest.main()
