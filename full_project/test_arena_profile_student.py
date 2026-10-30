import unittest
import numpy as np
import torch
from arena_profile_student import ProfileStudent,correction


class ProfileStudentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.net=ProfileStudent(np.random.default_rng(4).normal(size=(20,8)).astype('f')).eval()
        self.history=torch.tensor([[0,0,0],[1,2,3],[0,0,4]])
        self.age=torch.tensor([[0.,0.,0.],[4.,2.,1.],[0.,0.,1.]])

    def test_zero_start_and_empty_fallback(self):
        q,base=self.net(self.history,self.age)
        torch.testing.assert_close(q,base)
        torch.testing.assert_close(q[0],torch.zeros(8))

    def test_candidate_permutation_and_labels_do_not_enter(self):
        raw=dict(history=self.history.numpy(),age_hours=self.age.numpy(),items=np.arange(1,11)[None].repeat(3,0))
        raw['labels']=np.zeros((3,10));first=correction(self.net,raw,np.arange(3))
        with torch.no_grad():self.net.output[-1].weight.normal_(std=.02)
        p=np.arange(9,-1,-1);a=correction(self.net,raw,np.arange(3))
        changed=dict(raw,items=raw['items'][:,p],labels=np.ones((3,10)))
        np.testing.assert_allclose(correction(self.net,changed,np.arange(3)),a[:,p],atol=1e-7)
        np.testing.assert_array_equal(first,0)

    def test_dropout_and_cold_gradients_finite(self):
        q,_=self.net(self.history,self.age,True);q.square().sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.net.parameters()))

    def test_unknown_history_falls_back_and_augmentation_keeps_one(self):
        with torch.no_grad():self.net.items[5]=0;self.net.known_items[5]=False
        q,_=self.net(torch.tensor([[0,5,5]]),torch.zeros((1,3)))
        torch.testing.assert_close(q,torch.zeros_like(q))
        self.net.config=dict(self.net.config,history_dropout=1.)
        q,_=self.net(self.history,self.age,True)
        self.assertGreater(float(q[1].norm()),.9)


if __name__=='__main__':unittest.main()
