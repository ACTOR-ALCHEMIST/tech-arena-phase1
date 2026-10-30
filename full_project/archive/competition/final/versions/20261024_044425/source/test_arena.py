"""Acceptance invariants for sparse matching and event-level probability models."""
import unittest
import numpy as np
import torch
from scipy import sparse
from arena_core import co_features
from arena_event import aggregate,event_probability
from arena_finalize import blend
from arena_tabular import SlateMLP,normalizer
from arena_alignment import orthogonal_map


class ArenaInvariants(unittest.TestCase):
    def test_sparse_cf_matches_explicit_formula_including_duplicate_history(self):
        x=sparse.csr_matrix(np.array([[0,1,1,0],[0,1,1,1],[0,0,1,1]],dtype=np.float32))
        pop=np.asarray(x.sum(0)).ravel();c=(x.T@x).tocsr();c.setdiag(0);c.eliminate_zeros();c.sort_indices()
        items=np.array([[1,2,3,1,2,3,1,2,3,1]],np.int32)
        history=np.zeros((1,20),np.int32);history[0,:3]=[1,2,2]
        ages=np.zeros((1,20),np.float32);ages[0,:3]=[72,48,24]
        actual=co_features(c.indptr,c.indices,c.data,pop.astype(np.float32),items,history,ages)
        for j,item in enumerate(items[0]):
            expected=sum(float(c[item,h])/np.sqrt(pop[item]*pop[h]) for h in [1,2,2])
            self.assertAlmostEqual(float(actual[0,j,0]),expected,places=6)
        self.assertTrue(np.isfinite(actual).all())

    def test_event_gate_retains_no_click_mass_and_candidate_order(self):
        p=np.array([[0.01,0.2,0.05],[0.3,0.2,0.1]])
        gate=np.array([0.7,0.2])
        for temp in [0.5,1,2]:
            event=event_probability(p,gate,temp,1)
            np.testing.assert_allclose(event.sum(1),gate)
            np.testing.assert_array_equal(np.argsort(event,axis=1),np.argsort(p,axis=1))
            mixed=event_probability(p,gate,temp,0.5)
            self.assertTrue(((mixed>=0)&(mixed<=1)).all())

    def test_event_aggregation_and_blends_preserve_alignment(self):
        x=np.arange(60,dtype=np.float32).reshape(20,3)
        aggregated=aggregate(x)
        self.assertEqual(aggregated.shape,(2,9))
        np.testing.assert_allclose(aggregated[0,:3],x[:10].mean(0))
        a=np.array([[0.01,0.3],[0.1,0.2]]);b=np.array([[0.1,0.5],[0.2,0.4]])
        for kind in ["probability","logit"]:
            np.testing.assert_allclose(blend([a,b],[1,0],kind),a,atol=1e-12)
            mixed=blend([a,b],[0.5,0.5],kind)
            self.assertTrue(((mixed>=a)&(mixed<=b)).all())

    def test_slate_network_handles_no_click_and_roundtrips_weights(self):
        torch.manual_seed(7)
        x=np.random.default_rng(7).normal(size=(30,5)).astype(np.float32)
        mean,scale=normalizer(x)
        model=SlateMLP(5,mean,scale,'event_ce',dropout=0).eval()
        inputs=torch.tensor(x.reshape(3,10,5))
        probability=model.probabilities(inputs)
        self.assertEqual(tuple(probability.shape),(3,10))
        self.assertTrue(torch.isfinite(probability).all())
        self.assertTrue((probability.sum(1)<=1+1e-6).all())
        loss=torch.nn.functional.cross_entropy(model(inputs),torch.full((3,),10,dtype=torch.long))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        copy=SlateMLP(5,mean,scale,'event_ce',dropout=0).eval();copy.load_state_dict(model.state_dict())
        torch.testing.assert_close(copy.probabilities(inputs),probability)

    def test_constant_feature_policy_prevents_unseen_category_amplification(self):
        x=np.zeros((30,2),np.float32);x[:,0]=np.arange(30)
        mean,scale=normalizer(x,'zero')
        self.assertLess(abs(float((1-mean[1])/scale[1])),1e-10)

    def test_context_model_is_equivariant_to_candidate_permutation(self):
        torch.manual_seed(7)
        x=torch.randn(3,10,5)
        model=SlateMLP(5,np.zeros(5),np.ones(5),'event_ce',0,'context').eval()
        permutation=torch.randperm(10)
        torch.testing.assert_close(model.probabilities(x[:,permutation]),model.probabilities(x)[:,permutation])

    def test_alignment_recovers_rotation_and_preserves_matching_scores(self):
        rng=np.random.default_rng(7);a=rng.normal(size=(100,8));q,_=np.linalg.qr(rng.normal(size=(8,8)))
        fitted=orthogonal_map(a,a@q)
        np.testing.assert_allclose(fitted,q,atol=1e-6)
        u=rng.normal(size=(5,8));v=rng.normal(size=(5,8))
        np.testing.assert_allclose(np.sum((u@fitted)*(v@fitted),axis=1),np.sum(u*v,axis=1),atol=1e-6)


if __name__=="__main__":unittest.main()
