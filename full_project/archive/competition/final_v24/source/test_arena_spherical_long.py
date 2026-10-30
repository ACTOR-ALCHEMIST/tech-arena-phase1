import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arena_spherical_long import extra_features


class LongSphericalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        rng=np.random.default_rng(128);u=rng.normal(size=(7,8)).astype('f');v=rng.normal(size=(18,8)).astype('f')
        u[0]=v[0]=0
        for epoch,scale in [(64,1.),(128,.9)]:
            np.savez(self.folder/f'factors_e{epoch}.npz',users=u*scale,items=v*scale,bias=np.arange(18,dtype='f')*.01)
        self.data={'users':np.array([0,2,5]),'items':rng.integers(1,18,size=(3,10)),
                   'history':np.array([[0]*20,[1,2,3]+[0]*17,[4,5,6,7]+[0]*16]),
                   'age_hours':np.ones((3,20),dtype='f'),'timestamps':np.ones(3,dtype='i8')*1737849600,
                   'labels':np.zeros((3,10))}

    def tearDown(self):self.temp.cleanup()

    def test_two_fixed_views_are_finite_and_ignore_targets(self):
        with patch('arena_spherical_long.model',return_value=self.folder):
            a,names=extra_features(self.data,np.arange(3),'2025-01-26')
            self.data['labels'][:]=1
            b,other=extra_features(self.data,np.arange(3),'2025-01-26')
        self.assertEqual(a.shape,(30,44));self.assertEqual(len(set(names)),44)
        self.assertTrue(names[0].startswith('spherelong64_'));self.assertTrue(names[22].startswith('spherelong128_'))
        self.assertEqual(names,other);np.testing.assert_array_equal(a,b);self.assertTrue(np.isfinite(a).all())

    def test_candidate_permutation_equivariance(self):
        permutation=np.array([3,0,5,8,2,7,1,9,4,6])
        with patch('arena_spherical_long.model',return_value=self.folder):
            a,_=extra_features(self.data,np.arange(3),'2025-01-26')
            b,_=extra_features(dict(self.data,items=self.data['items'][:,permutation]),np.arange(3),'2025-01-26')
        np.testing.assert_allclose(a.reshape(3,10,44)[:,permutation],b.reshape(3,10,44),atol=2e-6,rtol=1e-5)

    def test_rejects_a_representation_newer_than_query(self):
        with self.assertRaises(AssertionError):extra_features(self.data,np.arange(3),'2025-01-27')


if __name__=='__main__':unittest.main()
