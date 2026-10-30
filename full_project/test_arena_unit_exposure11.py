import unittest
import numpy as np
import torch
from arena_unit_exposure11 import targets,global_features,globals_design,exposure_logits
from arena_conditional_sphere import ConditionalSphere,unit


class Exposure11Tests(unittest.TestCase):
    def test_no_click_target_and_negative_geometry_gradient(self):
        y=np.zeros((3,10),'i');y[1,3]=1
        np.testing.assert_array_equal(targets(y),[10,3,10])
        r=np.random.default_rng(1);rep=dict(users=unit(r.normal(size=(4,8))),items=unit(r.normal(size=(20,8))))
        net=ConditionalSphere(rep,(np.ones(4),np.ones(20)),(np.ones(4),np.ones(20)),3.,1.)
        z=exposure_logits(net,torch.tensor([1]),torch.arange(1,11)[None],torch.zeros((1,10)),3.)
        torch.nn.functional.cross_entropy(z,torch.tensor([10])).backward()
        self.assertGreater(float(net.users.weight.grad[1].norm()),1e-5)
        self.assertTrue(torch.isfinite(net.items.weight.grad).all())

    def test_outside_option_preserves_conditional_choice(self):
        inside=torch.tensor([[1.,2.,3.]])
        q=[]
        for outside in [-3.,4.]:
            p=torch.cat([inside,torch.tensor([[outside]])],1).softmax(1)[:,:3]
            q.append(p/p.sum(1,keepdim=True))
        torch.testing.assert_close(q[0],q[1])

    def test_global_inputs_ignore_labels_and_disable_constants(self):
        raw=dict(users=np.array([1,2]),context=np.zeros((2,6)),labels=np.zeros((2,10)))
        stats=dict(ub=np.array([0.,.2,.3]));a=global_features(raw,np.arange(2),stats)
        b=global_features(dict(raw,labels=np.ones((2,10))),np.arange(2),stats)
        np.testing.assert_array_equal(a,b)
        head=dict(mean=np.zeros(7),scale=np.ones(7),known=np.array([1,0,0,0,0,0,0]))
        np.testing.assert_array_equal(globals_design(np.ones((2,7)),head)[:,1:],0)


if __name__=='__main__':unittest.main()
