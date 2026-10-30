import unittest
import numpy as np
from arena_event_mass import features,apply,design


class EventMassTests(unittest.TestCase):
    def test_candidate_ratios_and_order_preserved(self):
        rng=np.random.default_rng(3);p=rng.random((5,10))*.08
        changed=apply(p,np.arange(5)*.05,.5)
        np.testing.assert_allclose(changed/changed.sum(1,keepdims=True),p/p.sum(1,keepdims=True),atol=1e-15)
        np.testing.assert_array_equal(changed.argmax(1),p.argmax(1))
        np.testing.assert_allclose(apply(p,np.zeros(5),1),p,atol=1e-15)

    def test_candidate_permutation_invariant_event_features(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(5,10,244));p=rng.random((5,10))*.05
        # Columns read from the first candidate are shared event covariates.
        from arena_event_mass import RECIPE
        for i in RECIPE['scalar_columns']:x[:,:,i]=x[:,0,i,None]
        order=rng.permutation(10)
        np.testing.assert_allclose(features(x,p),features(x[:,order],p[:,order]),atol=5e-7)

    def test_constant_features_disabled(self):
        f=np.array([[0.,1.],[3.,2.]])
        x=design(f,np.zeros(2),np.ones(2),np.array([0.,1.]))
        np.testing.assert_array_equal(x[:,1],0)


if __name__=='__main__':unittest.main()
