import unittest
import numpy as np
from arena_borderline import rerank


class BorderlineTests(unittest.TestCase):
    def test_preserves_outside_pair_and_event_mass(self):
        p=np.array([[.2,.19,.04,.03,.02,.01,.01,.01,.01,.01]])
        d=np.arange(10)[None]/10
        q=rerank(p,d,1)
        np.testing.assert_array_equal(q[:,2:],p[:,2:])
        np.testing.assert_allclose(q.sum(1),p.sum(1),atol=1e-15)
        self.assertGreater(q[0,1],p[0,1])

    def test_clear_winner_and_zero_residual_are_unchanged(self):
        p=np.array([[.5,.01,.01,.01,.01,.01,.01,.01,.01,.01]])
        np.testing.assert_array_equal(rerank(p,np.ones_like(p),1),p)
        p[0,1]=.49
        np.testing.assert_allclose(rerank(p,np.zeros_like(p),1),p,atol=1e-15)

    def test_permutation_equivariant_without_ties(self):
        rng=np.random.default_rng(8);p=rng.random((10,10))*.06;d=rng.normal(size=p.shape)
        permutation=rng.permutation(10)
        np.testing.assert_allclose(rerank(p[:,permutation],d[:,permutation],.5),rerank(p,d,.5)[:,permutation],atol=1e-15)


if __name__=='__main__':unittest.main()
