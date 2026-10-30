import unittest
import numpy as np
from arena_metric_geometry import cosine,preference_probabilities


class LatentDeltaTests(unittest.TestCase):
    def test_rotation_permutation_and_click_mass(self):
        rng=np.random.default_rng(826);u=rng.normal(size=(8,8));v=rng.normal(size=(30,8));u[0]=v[0]=0
        raw={'users':np.arange(8),'items':rng.integers(1,30,size=(8,10)),'labels':np.zeros((8,10))}
        rep=dict(users=u,items=v);before=cosine(rep,raw,np.arange(8))
        q,_=np.linalg.qr(rng.normal(size=(8,8)));perm=rng.permutation(10)
        after=cosine(dict(users=u@q,items=v@q),dict(raw,items=raw['items'][:,perm]),np.arange(8))
        np.testing.assert_allclose(before[:,perm],after,atol=5e-16,rtol=2e-14)
        p=rng.dirichlet(np.ones(11),size=8)[:,:10]
        out=preference_probabilities(p,before,.5)
        np.testing.assert_allclose(out.sum(1),p.sum(1),atol=2e-16)
        self.assertTrue(np.isfinite(out).all());np.testing.assert_array_equal(before[0],0)

    def test_query_targets_and_cached_tables_unchanged(self):
        rng=np.random.default_rng(26);rep=dict(users=rng.normal(size=(7,8)),items=rng.normal(size=(25,8)))
        raw=dict(users=np.arange(7),items=rng.integers(1,25,size=(7,10)),labels=np.zeros((7,10)))
        u,v=rep['users'].copy(),rep['items'].copy();a=cosine(rep,raw,np.arange(7));raw['labels'][:]=1
        b=cosine(rep,raw,np.arange(7));np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(rep['users'],u);np.testing.assert_array_equal(rep['items'],v)


if __name__=='__main__':unittest.main()
