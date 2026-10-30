import unittest
import numpy as np
from scipy import sparse
from arena_ppmi import ppmi_graph
from arena_core import clear_diagonal


class PPMITests(unittest.TestCase):
    def test_symmetric_finite_and_unseen_items_zero(self):
        x = sparse.csr_matrix(np.array([[0, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 0]], 'float32'))
        c = x.T@x
        c = clear_diagonal(c)
        pop = np.asarray(x.sum(0)).ravel().astype('float64')
        g = ppmi_graph(c, pop, 3, shift=.5, shrink=3)
        np.testing.assert_allclose(g.toarray(), g.T.toarray(), atol=1e-14)
        self.assertTrue(np.isfinite(g.data).all())
        self.assertTrue((g.data >= 0).all())
        np.testing.assert_array_equal(g.getrow(4).toarray(), 0)
        np.testing.assert_array_equal(g.diagonal(), 0)

    def test_small_example_matches_dense_formula(self):
        c = sparse.csr_matrix(np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]], 'float64'))
        pop = np.array([3., 4., 5.])
        actual = ppmi_graph(c, pop, 20, shift=2, shrink=3).toarray()
        dense = c.toarray()
        expected = np.maximum(np.log(np.maximum(dense, 1e-12)*20/(2*pop[:, None]*pop[None])), 0)*dense/(dense+3)
        np.fill_diagonal(expected, 0)
        scale = 1/np.sqrt((expected.sum(1)+20).clip(1e-12))
        expected = scale[:, None]*expected*scale[None]
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_diagonal_removal_preserves_every_other_sparse_entry(self):
        rng = np.random.default_rng(93)
        x = sparse.random(11, 17, density=.4, random_state=rng, format='csr')
        counts = (x.T@x).tocsr()
        expected = counts.toarray()
        np.fill_diagonal(expected, 0)
        actual = clear_diagonal(counts)
        np.testing.assert_array_equal(actual.toarray(), expected)
        # Original untouched and no missing diagonal assumptions for padding0.
        self.assertTrue(np.any(counts.diagonal() > 0))
        self.assertTrue(actual.has_sorted_indices)


if __name__ == '__main__':
    unittest.main()
