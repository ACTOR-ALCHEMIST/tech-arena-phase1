import unittest
import numpy as np
from arena_unit_consensus import from_representations


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(1)
        self.reps=[]
        for _ in range(3):
            rep={k:self.rng.normal(size=shape).astype('f') for k,shape in [('users',(5,8)),('items',(20,8)),('bias',(20,))]}
            for k in ['users','items']:rep[k][0]=0
            self.reps.append(rep)
        self.raw=dict(users=np.array([1,2]),items=np.arange(1,11)[None].repeat(2,0),
                      history=np.array([[0,1,2],[0,0,0]]),age_hours=np.array([[0,3,1],[0,0,0]],dtype='f'))

    def test_independent_basis_rotations_and_label_changes(self):
        a,n=from_representations(self.raw,np.arange(2),*self.reps);rot=[]
        for rep in self.reps:
            q=np.linalg.qr(self.rng.normal(size=(8,8)))[0]
            rot.append(dict(rep,users=rep['users']@q,items=rep['items']@q))
        changed=dict(self.raw,labels=np.ones((2,10)))
        b,m=from_representations(changed,np.arange(2),*rot)
        self.assertEqual(n,m);np.testing.assert_allclose(a,b,atol=2e-6)

    def test_permutation_and_zero_history_are_finite(self):
        a,n=from_representations(self.raw,np.arange(2),*self.reps);p=np.arange(9,-1,-1)
        b,m=from_representations(dict(self.raw,items=self.raw['items'][:,p]),np.arange(2),*self.reps)
        np.testing.assert_allclose(b.reshape(2,10,31),a.reshape(2,10,31)[:,p],atol=2e-6)
        self.assertEqual(len(n),31);self.assertTrue(np.isfinite(a).all())


if __name__=='__main__':unittest.main()
