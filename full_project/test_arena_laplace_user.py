import unittest
import numpy as np
import torch
from arena_laplace_user import predictive_uncertainty, features_for_posterior
from arena_bayes_mf import BayesianMF, ANCILLARY


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(101)
        self.p = rng.uniform(.01,.1,(4,10))
        self.z = rng.normal(size=(4,10,3))
        a = rng.normal(size=(4,3,3))
        self.cov = np.einsum('bik,bjk->bij',a,a)*.02

    def test_zero_curvature_uncertainty_recovers_map(self):
        a,b,p,std = predictive_uncertainty(self.p,self.z,np.zeros_like(self.cov))
        np.testing.assert_allclose(p,self.p,atol=1e-15)
        np.testing.assert_array_equal(a,0)
        np.testing.assert_array_equal(b,0)
        np.testing.assert_array_equal(std,0)

    def test_rotations_preserve_all_uncertainty_features(self):
        rng=np.random.default_rng(72)
        r,_=np.linalg.qr(rng.normal(size=(3,3)))
        expected=predictive_uncertainty(self.p,self.z,self.cov)
        actual=predictive_uncertainty(self.p,self.z@r,np.einsum('ij,bjk,kl->bil',r.T,self.cov,r))
        for a,b in zip(expected,actual):np.testing.assert_allclose(a,b,atol=1e-12)

    def test_candidate_permutation_and_batch_partition(self):
        perm=np.array([9,4,3,2,8,0,7,1,6,5])
        expected=predictive_uncertainty(self.p,self.z,self.cov)
        actual=predictive_uncertainty(self.p[:,perm],self.z[:,perm],self.cov)
        first=predictive_uncertainty(self.p[:1],self.z[:1],self.cov[:1])
        rest=predictive_uncertainty(self.p[1:],self.z[1:],self.cov[1:])
        for i,(a,b) in enumerate(zip(expected,actual)):
            np.testing.assert_allclose(a[:,perm],b,atol=1e-12)
            np.testing.assert_allclose(a,np.concatenate([first[i],rest[i]]),atol=1e-12)

    def test_query_features_do_not_read_targets(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(33)
        u,v=rng.normal(size=(4,3)).astype('f'),rng.normal(size=(12,3)).astype('f')
        u[0]=v[0]=0
        ancillary={k:np.ones(4 if k in ['ub','u_precision','bu_precision','head_users'] else 12,'f') for k in ANCILLARY}
        net=BayesianMF(dict(users=u,items=v),ancillary,dict(fine_items=False)).eval()
        with torch.no_grad():net.users.weight[1].add_(.05)
        q=dict(users=np.array([0,1,2],np.int32),items=rng.integers(0,12,(3,10),dtype=np.int32),
               history=rng.integers(0,12,(3,20),dtype=np.int32),age_hours=rng.uniform(0,120,(3,20)).astype('f'),context=rng.normal(size=(3,6)).astype('f'))
        covariance=np.tile(np.eye(4,dtype='f')*.2,(4,1,1))
        a,names=features_for_posterior(net,covariance,q,np.arange(3))
        b,other=features_for_posterior(net,covariance,dict(q,labels=np.ones((3,10))),np.arange(3))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(30,36))
        self.assertEqual(names,other)
        self.assertEqual(len(set(names)),36)
        self.assertTrue(np.isfinite(a).all())


if __name__=='__main__':unittest.main()
