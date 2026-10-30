import unittest
import numpy as np
import torch
from arena_history_data import GeometryBank,TOKEN_WIDTH
from arena_history_model import HistoryReranker,CONFIGS


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(8388);torch.manual_seed(8388);torch.set_num_threads(2)
        self.raw=dict(users=np.array([0,1,2,3]),items=self.rng.integers(1,25,size=(4,10)),
                      history=np.array([[0]*20,[1,2,3]+[0]*17,[3,4]+[0]*18,[5,6,7,8]+[0]*16]),
                      age_hours=np.tile(np.arange(20,0,-1,dtype='f'),(4,1)),timestamps=np.full(4,1737849600,dtype='i8'),labels=np.zeros((4,10)))
        self.reps=[]
        for _ in range(3):
            u=self.rng.normal(size=(7,8)).astype('f');v=self.rng.normal(size=(25,8)).astype('f');u[0]=v[0]=0
            self.reps.append(dict(users=u,items=v))
        self.p=torch.tensor(self.rng.dirichlet(np.ones(11),size=4)[:,:10],dtype=torch.float32)
        self.x=torch.randn(4,10,244)

    def render(self,raw=None,reps=None):
        return GeometryBank(raw or self.raw,{'2025-01-26':reps or self.reps}).render('2025-01-26',np.arange(4),self.p.numpy())

    def test_tokens_ignore_targets_and_independent_basis_rotations(self):
        a,m,pair=self.render();self.assertEqual(a.shape,(4,10,20,TOKEN_WIDTH))
        self.raw['labels'][:]=1;b,_,_=self.render();np.testing.assert_array_equal(a,b)
        rotated=[]
        for r in self.reps:
            q,_=np.linalg.qr(self.rng.normal(size=(8,8)))
            rotated.append({k:(v@q).astype('f') for k,v in r.items()})
        b,other,p=self.render(reps=rotated)
        np.testing.assert_allclose(a,b,atol=3e-6,rtol=1e-5);np.testing.assert_array_equal(m,other)
        np.testing.assert_allclose(pair,p,atol=3e-6,rtol=1e-5)

    def test_zero_start_and_empty_history_fallback(self):
        t,m,pair=self.render();args=[self.x,self.p,torch.tensor(t),torch.tensor(m),torch.tensor(pair)]
        for cfg in CONFIGS.values():
            net=HistoryReranker(np.zeros(244),np.ones(244),cfg).eval()
            torch.testing.assert_close(net.probabilities(*args),self.p,atol=1e-7,rtol=1e-6)
            torch.nn.init.normal_(net.score.weight,std=.2)
            np.testing.assert_array_equal(net.corrections(*args).detach().numpy()[0],0)
            torch.testing.assert_close(net.probabilities(*args).sum(1),self.p.sum(1),atol=2e-7,rtol=1e-6)

    def test_trained_model_and_tokens_are_candidate_equivariant(self):
        perm=np.array([2,1,0,5,8,3,7,4,9,6]);t,m,pair=self.render()
        raw=dict(self.raw,items=self.raw['items'][:,perm])
        t2,m2,pair2=GeometryBank(raw,{'2025-01-26':self.reps}).render('2025-01-26',np.arange(4),self.p.numpy()[:,perm])
        np.testing.assert_allclose(t[:,perm],t2,atol=3e-6)
        net=HistoryReranker(np.zeros(244),np.ones(244),CONFIGS['history_din_slate']).eval()
        torch.nn.init.normal_(net.score.weight,std=.2)
        a=net.corrections(self.x,self.p,torch.tensor(t),torch.tensor(m),torch.tensor(pair))
        b=net.corrections(self.x[:,perm],self.p[:,perm],torch.tensor(t2),torch.tensor(m2),torch.tensor(pair2))
        torch.testing.assert_close(a[:,perm],b,atol=2e-6,rtol=1e-5)

    def test_gradients_finite_with_empty_and_short_histories(self):
        t,m,pair=self.render();net=HistoryReranker(np.zeros(244),np.ones(244),CONFIGS['history_din_slate'])
        z=net(self.x,self.p,torch.tensor(t),torch.tensor(m),torch.tensor(pair))
        torch.nn.functional.cross_entropy(z,torch.tensor([10,2,4,6])).backward()
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_future_geometry_rejected(self):
        with self.assertRaises(AssertionError):GeometryBank(self.raw).render('2025-01-27',np.arange(4),self.p.numpy())


if __name__=='__main__':unittest.main()
