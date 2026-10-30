"""Useful invariants for OOF residual reranking and slate relations."""
import unittest
import numpy as np
import torch
from arena_two_stage import Reranker,CONFIGS,score_features
from arena_two_stage_data import day_indices,raw_inputs,relation_features


class TwoStageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(838);torch.set_num_threads(2)
        self.x=torch.randn(4,10,244);self.x[:,:,25]=.5
        self.p=torch.randn(4,11).softmax(1)[:,:10]
        v=torch.nn.functional.normalize(torch.randn(4,10,8),dim=2)
        self.pair=v@v.transpose(1,2)

    def test_zero_residual_preserves_probabilities(self):
        for name,cfg in CONFIGS.items():
            model=Reranker(np.zeros(244),np.ones(244),cfg).eval()
            torch.testing.assert_close(model.probabilities(self.x,self.p,self.pair),self.p,atol=6e-8,rtol=1e-6)

    def test_trained_transformer_is_candidate_permutation_equivariant(self):
        model=Reranker(np.zeros(244),np.ones(244),CONFIGS['edge_transformer']).eval()
        torch.nn.init.normal_(model.rank.weight,std=.2);torch.nn.init.normal_(model.propensity.weight,std=.2)
        perm=torch.randperm(10)
        a=model.probabilities(self.x,self.p,self.pair)
        b=model.probabilities(self.x[:,perm],self.p[:,perm],self.pair[:,perm][:,:,perm])
        torch.testing.assert_close(a[:,perm],b,atol=1e-7,rtol=1e-6)

    def test_tie_gate_focuses_uncertain_slates(self):
        p=torch.tensor([[.4,.4]+[.01]*8,[.7,.1]+[.01]*8])
        pair=torch.eye(10)[None].expand(2,-1,-1)
        _,_,gate,_=score_features(p,pair,torch.ones(2))
        self.assertGreater(float(gate[0]),float(gate[1])+.5)

    def test_extreme_cold_slates_have_finite_gradients(self):
        model=Reranker(np.zeros(244),np.ones(244),CONFIGS['edge_transformer'])
        x=torch.zeros(4,10,244);p=torch.ones(4,10)*1e-7;pair=torch.zeros(4,10,10)
        z=model(x,p,pair);torch.nn.functional.cross_entropy(z,torch.tensor([10,0,5,2])).backward()
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_day_slice_keeps_midnight_boundary_and_excludes_targets(self):
        from run import timestamp
        t=timestamp('2025-01-24');data={'timestamps':np.array([t-1,t,t+86399,t+86400]),'labels':np.ones((4,10))}
        np.testing.assert_array_equal(day_indices(data,'2025-01-24'),[1,2])
        self.assertNotIn('labels',raw_inputs(data))

    def test_relations_ignore_labels_and_latent_rotation(self):
        rng=np.random.default_rng(83);rep={'items':rng.normal(size=(30,8)).astype('f')}
        data={'items':rng.integers(1,30,size=(5,10)),'labels':rng.integers(0,2,size=(5,10))}
        before=relation_features(rep,data,np.arange(5));data['labels'][:]=1-data['labels']
        q,_=np.linalg.qr(rng.normal(size=(8,8)))
        after=relation_features({'items':(rep['items']@q).astype('f')},data,np.arange(5))
        np.testing.assert_allclose(before,after,atol=4e-7,rtol=1e-6)

    def test_utility_transfer_zero_identity_and_full_replacement(self):
        from arena_two_stage_campaign import compose
        rng=np.random.default_rng(44)
        a=rng.dirichlet(np.ones(11),size=8)[:,:10]
        b=rng.dirichlet(np.ones(11),size=8)[:,:10]
        np.testing.assert_allclose(compose(a,a,b,'transfer',0),a,atol=2e-16)
        np.testing.assert_allclose(compose(a,a,b,'transfer',1),b,atol=2e-16)

    def test_choice_only_correction_preserves_event_click_mass(self):
        model=Reranker(np.zeros(244),np.ones(244),CONFIGS['origin_choice_only']).eval()
        torch.nn.init.normal_(model.rank.weight,std=.4)
        pred=model.probabilities(self.x,self.p,self.pair)
        torch.testing.assert_close(pred.sum(1),self.p.sum(1),atol=1e-7,rtol=1e-6)
        self.assertGreater(float((pred-self.p).abs().max()),1e-4)


if __name__=='__main__':unittest.main()
