import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arena_aligned_repr import rotation, features, model, RECIPE, TAG, LazyAlignedRows


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(154)
        u, v = rng.normal(size=(6, 8)).astype('float32'), rng.normal(size=(15, 8)).astype('float32')
        u[0] = v[0] = 0
        self.rep = dict(users=u, items=v, known_users=np.linalg.norm(u, axis=1)>0, known_items=np.linalg.norm(v, axis=1)>0)
        self.q = dict(users=np.array([0, 1, 2]), items=rng.integers(0, 15, (3, 10)),
                      history=np.array([[0]*20, [1, 2, 0]+[0]*17, [3, 4, 5, 6]+[0]*16]),
                      age_hours=np.tile(np.arange(20, dtype='float32'), (3, 1)))

    def test_recovers_rotation_and_preserves_preferences(self):
        rng = np.random.default_rng(46)
        anchor, users = rng.normal(size=(100, 8)), rng.normal(size=(20, 8))
        r, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        fresh, uf = anchor@r, users@r
        transform = rotation(fresh, anchor, rng.uniform(.1, 20, 100))
        np.testing.assert_allclose(fresh@transform, anchor, atol=1e-12)
        np.testing.assert_allclose((uf@transform)@(fresh@transform).T, uf@fresh.T, atol=1e-12)

    def test_queries_need_no_labels_and_cold_rows_finite(self):
        a, names = features(self.rep, self.q, np.arange(3))
        b, _ = features(self.rep, dict(self.q, labels=np.ones((3, 10))), np.arange(3))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (30, 75))
        self.assertEqual(len(set(names)), 75)
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a[:10, :8], 0)

    def test_candidate_permutation_is_equivariant(self):
        permutation = np.array([9, 2, 7, 1, 8, 3, 0, 5, 6, 4])
        a, _ = features(self.rep, self.q, np.arange(3))
        q = dict(self.q, items=self.q['items'][:, permutation])
        b, _ = features(self.rep, q, np.arange(3))
        np.testing.assert_array_equal(a.reshape(3, 10, 75)[:, permutation], b.reshape(3, 10, 75))

    def test_final_cache_needs_no_reference_snapshot_or_training(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/TAG).mkdir()
            np.savez(path/TAG/'factors.npz', **self.rep)
            (path/TAG/'result.json').write_text(json.dumps(dict(recipe=RECIPE)))
            with patch('arena_aligned_repr.snapshot', return_value=path), patch('arena_spherical_repr.model', side_effect=AssertionError('must not train')):
                cached = model('2025-01-27')
            np.testing.assert_array_equal(cached['items'], self.rep['items'])

    def test_lazy_rows_exactly_match_eager_for_slices_and_reordered_rows(self):
        indices = np.array([2, 0, 1])
        eager, _ = features(self.rep, self.q, indices)
        lazy = LazyAlignedRows(self.rep, self.q, indices)
        for selection in [slice(None), slice(2, 23, 3), slice(None, None, -1), np.array([29, 0, 13, 13, 2]), -1, 11]:
            np.testing.assert_array_equal(lazy[selection], eager[selection])
        self.assertEqual(lazy.shape, eager.shape)
        self.assertEqual(lazy[[]].shape, (0, 75))


if __name__ == '__main__':
    unittest.main()
