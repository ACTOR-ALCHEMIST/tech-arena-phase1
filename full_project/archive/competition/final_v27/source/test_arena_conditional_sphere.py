import unittest
import numpy as np
import torch
from arena_conditional_sphere import ConditionalSphere,covariates,unit
from arena_metric_geometry import preference_probabilities


class ConditionalSphereTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.rep=dict(users=unit(rng.normal(size=(4,3))),items=unit(rng.normal(size=(15,3))))
        for v in self.rep.values():v[0]=0
        self.raw=dict(users=np.array([1,2]),items=np.arange(1,11)[None].repeat(2,0),
                      history=np.array([[0,1,2],[0,0,0]]),age_hours=np.array([[0,2,1],[0,0,0]],dtype='f'),
                      labels=np.zeros((2,10)))
        self.ib=np.linspace(-1,1,15)
        self.net=ConditionalSphere(self.rep,(np.ones(4),np.ones(15)),(np.ones(4),np.ones(15)),6.,1.)

    def test_label_independence_rotation_and_empty_history(self):
        ix=np.arange(2);x=covariates(self.raw,ix,self.rep,self.ib)
        q=np.linalg.qr(np.random.default_rng(4).normal(size=(3,3)))[0]
        rotated={k:v@q for k,v in self.rep.items()}
        changed=dict(self.raw,labels=np.ones((2,10)))
        np.testing.assert_allclose(x,covariates(changed,ix,rotated,self.ib),atol=4e-7)
        np.testing.assert_array_equal(x[1,:,2:],0)

    def test_candidate_permutation_and_zero_correction_mass(self):
        u=torch.tensor([1,2]);ids=torch.tensor(self.raw['items']);off=torch.randn(2,10)
        p=torch.randperm(10)
        torch.testing.assert_close(self.net(u,ids[:,p],off[:,p]),self.net(u,ids,off)[:,p])
        baseline=np.random.default_rng(1).random((2,10))*.1
        replay=preference_probabilities(baseline,np.zeros((2,10)),1)
        np.testing.assert_allclose(replay,baseline,atol=1e-15)

    def test_prior_occurrence_normalization(self):
        counts=(np.array([1,3,1,1]),np.ones(15))
        net=ConditionalSphere(self.rep,counts,(np.ones(4),np.ones(15)),6.,2.)
        with torch.no_grad():net.users.weight[1]+=torch.tensor([.2,-.1,.3])
        uid=torch.tensor([1,1,1]);ids=torch.zeros((3,10),dtype=torch.long)
        expected=(net.lookup(torch.tensor([1]),True)-net.u0[[1]]).square().sum()
        torch.testing.assert_close(net.penalty(uid,ids)*3,expected)

    def test_unknown_and_cold_gradients_finite(self):
        u=torch.tensor([0,1]);ids=torch.tensor(self.raw['items']);ids[0]=0
        z=self.net(u,ids,torch.zeros((2,10)))
        loss=torch.nn.functional.cross_entropy(z,torch.tensor([0,2]))+self.net.penalty(u,ids)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.net.parameters()))
        np.testing.assert_array_equal(self.net.representation()['users'][0],0)


if __name__=='__main__':unittest.main()
