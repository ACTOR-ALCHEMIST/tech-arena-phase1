import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from arena_v3_model import HistoryTransformer


class HistoryTests(unittest.TestCase):
    def make(self,unfreeze=False):
        torch.manual_seed(9)
        config=dict(width=16,heads=2,history_layers=1,slate_layers=1,hidden=[32,16],context_width=8,dropout=0.,foundation_dims=4,
            base_config=dict(architecture='v2_wide',hidden=[32,16],context_width=8,loss='event_ce',dropout=0.))
        config['unfreeze_base']=unfreeze
        model=HistoryTransformer(config)
        history=torch.randn(3,20,36);history[:,:,-1]=1;history[0]=0
        inputs=[torch.randn(3,374),history,torch.randn(3,10,144),torch.randn(3,10,4)]
        return model,inputs

    def test_initial_base_exact_empty_and_gradients(self):
        model,inputs=self.make();model.eval()
        with torch.no_grad():
            actual=model(*inputs);expected=model.base(inputs[2][:,:,:112])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(actual).all())
        loss=torch.nn.functional.cross_entropy(model(*inputs),torch.tensor([10,3,7]));loss.backward()
        self.assertGreater(float(model.score.weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))

    def test_nonzero_attention_permutation_and_reload(self):
        model,inputs=self.make();torch.nn.init.normal_(model.score.weight,std=.03);model.eval()
        order=torch.tensor([4,1,7,0,9,5,3,8,2,6]);shuffled=[inputs[0],inputs[1],inputs[2][:,order],inputs[3][:,order]]
        with torch.no_grad():a=model.probabilities(*inputs);b=model.probabilities(*shuffled)
        torch.testing.assert_close(a[:,order],b,rtol=1e-5,atol=1e-6)
        self.assertTrue((a.sum(1)<1).all())
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'model.pt';torch.save(model.state_dict(),path)
            fresh,_=self.make();fresh.load_state_dict(torch.load(path,weights_only=True));fresh.eval()
            with torch.no_grad():torch.testing.assert_close(fresh.probabilities(*inputs),a,rtol=0,atol=0)

    def test_larger_variant_really_updates_base(self):
        model,inputs=self.make(True)
        loss=torch.nn.functional.cross_entropy(model(*inputs),torch.tensor([10,3,7]));loss.backward()
        self.assertGreater(float(model.base.score.weight.grad.norm()),0)


if __name__=='__main__':unittest.main()
