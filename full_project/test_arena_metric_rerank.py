import unittest
import numpy as np
import torch
from arena_metric_rerank import profiles,MetricCorrection,preference_probabilities


class MetricRerankTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.rng=np.random.default_rng(8389)

    def test_rotated_frame_and_metric_give_identical_corrections(self):
        q,_=np.linalg.qr(self.rng.normal(size=(8,8)))
        p=self.rng.normal(size=(5,3,8));v=self.rng.normal(size=(5,10,8));a=self.rng.normal(size=(3,8,8))
        m=MetricCorrection().double();m.matrix.data.copy_(torch.from_numpy(a))
        n=MetricCorrection().double();n.matrix.data.copy_(torch.from_numpy(np.einsum('ad,kde,eb->kab',q.T,a,q)))
        x=m(torch.from_numpy(p),torch.from_numpy(v)).detach().numpy()
        y=n(torch.from_numpy(p@q),torch.from_numpy(v@q)).detach().numpy()
        np.testing.assert_allclose(x,y,atol=8e-14,rtol=1e-12)
        self.assertAlmostEqual(float(m.matrix.square().sum().detach()),float(n.matrix.square().sum().detach()),places=10)

    def test_geometry_excludes_query_labels_and_handles_cold_ids(self):
        rep={'users':self.rng.normal(size=(7,8)).astype('f'),'items':self.rng.normal(size=(15,8)).astype('f')}
        rep['users'][0]=0;rep['items'][0]=0
        raw={'users':np.array([0,2]),'items':self.rng.integers(0,15,size=(2,10)),
             'history':np.array([[0]*20,[1,2,3]+[0]*17]),'age_hours':np.ones((2,20),dtype='f'),'labels':np.zeros((2,10))}
        a,b=profiles(raw,np.arange(2),rep);raw['labels'][:]=1
        c,d=profiles(raw,np.arange(2),rep)
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
        self.assertTrue(np.isfinite(a).all());np.testing.assert_array_equal(a[0],0)

    def test_candidate_permutation_and_click_mass_conservation(self):
        p=self.rng.dirichlet(np.ones(11),size=6)[:,:10];delta=self.rng.normal(size=(6,10));perm=self.rng.permutation(10)
        a=preference_probabilities(p,delta,.5);b=preference_probabilities(p[:,perm],delta[:,perm],.5)
        np.testing.assert_allclose(a[:,perm],b,atol=2e-16)
        np.testing.assert_allclose(a.sum(1),p.sum(1),atol=2e-16)
        np.testing.assert_allclose(preference_probabilities(p,delta,0),p,atol=2e-16)

    def test_numpy_inference_matches_training_model(self):
        from arena_metric_geometry import correction
        m=MetricCorrection();a=self.rng.normal(size=(3,8,8)).astype('f')*.1
        m.matrix.data.copy_(torch.from_numpy(a))
        p=self.rng.normal(size=(17,3,8)).astype('f');v=self.rng.normal(size=(17,10,8)).astype('f')
        actual=correction(p,v,a)
        expected=m(torch.from_numpy(p),torch.from_numpy(v)).detach().numpy()
        np.testing.assert_allclose(actual,expected,atol=8e-7,rtol=1e-5)


if __name__=='__main__':unittest.main()
