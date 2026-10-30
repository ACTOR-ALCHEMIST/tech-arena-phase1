import unittest
import numpy as np
from arena_choice_gate import mix,event_gate


class ChoiceGateTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(83825)
        self.a=rng.dirichlet(np.ones(11),size=15)[:,:10]
        self.b=rng.dirichlet(np.ones(11),size=15)[:,:10]
        self.h=rng.integers(0,30,size=(15,20));self.h[:3]=0

    def test_mass_preservation_for_every_gate_and_kind(self):
        for gate in ['all','near_tie','smooth_tie','short_history','disagreement','confident_disagreement']:
            for kind in ['choice_log','choice_probability']:
                p=mix(self.a,self.b,self.h,gate,.5,kind)
                np.testing.assert_allclose(p.sum(1),self.a.sum(1),atol=3e-16)
                self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>=0)&(p<=1)).all())

    def test_zero_weight_and_candidate_permutation(self):
        perm=np.array([2,5,1,8,3,9,4,0,7,6])
        np.testing.assert_array_equal(mix(self.a,self.b,self.h,'all',0),self.a)
        for gate in ['all','near_tie','smooth_tie','short_history','disagreement','confident_disagreement']:
            a=mix(self.a,self.b,self.h,gate,.25)
            b=mix(self.a[:,perm],self.b[:,perm],self.h,gate,.25)
            np.testing.assert_allclose(a[:,perm],b,atol=2e-16)

    def test_tied_maxima_do_not_introduce_candidate_position_bias(self):
        a=np.array([[.1]*10,[.2,.2]+[.02]*8])
        b=np.array([[.01,.3]+[.01]*8,[.01,.25]+[.01]*8]);h=np.zeros((2,20))
        perm=np.array([1,0,2,3,4,5,6,7,8,9])
        for gate in ['disagreement','confident_disagreement']:
            np.testing.assert_array_equal(event_gate(a,b,h,gate),event_gate(a[:,perm],b[:,perm],h,gate))


if __name__=='__main__':unittest.main()
