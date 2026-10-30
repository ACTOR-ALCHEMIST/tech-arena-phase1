import unittest
import numpy as np
from arena_oof_user_rerank import probabilities


class OOFUserRerankTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(838)
        self.p=self.rng.dirichlet(np.ones(11),size=6)[:,:10]
        self.raw={'users':np.arange(6),'items':self.rng.integers(0,30,size=(6,10)),'labels':np.zeros((6,10))}
        self.v=self.rng.normal(size=(30,8));self.delta=self.rng.normal(size=(6,9))*.1
        self.ix=np.arange(6)

    def test_zero_mass_identity_and_choice_mass_conservation(self):
        for mode in ['joint','choice']:
            np.testing.assert_allclose(probabilities(self.p,self.raw,self.ix,self.delta,self.v,mode,0),self.p,atol=3e-16)
        out=probabilities(self.p,self.raw,self.ix,self.delta,self.v,'choice',1)
        np.testing.assert_allclose(out.sum(1),self.p.sum(1),atol=3e-16)

    def test_rotation_and_candidate_permutation(self):
        a=probabilities(self.p,self.raw,self.ix,self.delta,self.v,'joint',.5)
        q,_=np.linalg.qr(self.rng.normal(size=(8,8)));d=self.delta.copy();d[:,:8]=d[:,:8]@q
        perm=self.rng.permutation(10);raw=dict(self.raw,items=self.raw['items'][:,perm])
        b=probabilities(self.p[:,perm],raw,self.ix,d,self.v@q,'joint',.5)
        np.testing.assert_allclose(a[:,perm],b,atol=2e-16)

    def test_current_query_labels_do_not_affect_predictions(self):
        a=probabilities(self.p,self.raw,self.ix,self.delta,self.v,'joint',1)
        self.raw['labels'][:]=1
        b=probabilities(self.p,self.raw,self.ix,self.delta,self.v,'joint',1)
        np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
