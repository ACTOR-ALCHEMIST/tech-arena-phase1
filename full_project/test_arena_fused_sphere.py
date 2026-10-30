import unittest
import numpy as np
from arena_fused_sphere import fuse


class FusedSphereTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(2583)
        u=self.rng.normal(size=(12,8)).astype('f');v=self.rng.normal(size=(30,8)).astype('f')
        u[0]=v[0]=0
        self.a=dict(users=u,items=v,bias=self.rng.normal(size=30).astype('f'))
        self.known=np.ones(12,bool);self.known[0]=False
        self.pop=np.ones(30);self.pop[0]=0

    def test_recovers_independent_frame_for_identical_geometry(self):
        q,_=np.linalg.qr(self.rng.normal(size=(8,8)))
        b=dict(users=(self.a['users']@q).astype('f'),items=(self.a['items']@q).astype('f'))
        result=fuse(self.a,b,self.known,self.pop,.5)
        u=self.a['users']/np.linalg.norm(self.a['users'],axis=1,keepdims=True).clip(1e-8)
        v=self.a['items']/np.linalg.norm(self.a['items'],axis=1,keepdims=True).clip(1e-8)
        np.testing.assert_allclose(result['users'],u,atol=2e-7,rtol=1e-5)
        np.testing.assert_allclose(result['items'],v,atol=2e-7,rtol=1e-5)

    def test_cold_ids_and_agreement_norms(self):
        b=dict(users=self.rng.normal(size=(12,8)).astype('f'),items=self.rng.normal(size=(30,8)).astype('f'))
        self.known[4]=False;self.pop[7]=0
        result=fuse(self.a,b,self.known,self.pop,.75)
        np.testing.assert_array_equal(result['users'][[0,4]],0)
        np.testing.assert_array_equal(result['items'][[0,7]],0)
        np.testing.assert_array_equal(result['bias'][[0,7]],0)
        self.assertTrue(np.isfinite(result['users']).all())
        self.assertLessEqual(float(np.linalg.norm(result['items'],axis=1).max()),1.000001)
        self.assertLess(float(np.linalg.norm(result['items'][1])),1.)


if __name__=='__main__':unittest.main()
