import unittest
import numpy as np
import torch
from arena_robust_mf import RobustMF


class RobustMFTest(unittest.TestCase):
    def setup_model(self, fine_tune=False, **kwargs):
        torch.manual_seed(517)
        rng = np.random.default_rng(517)
        u = rng.normal(scale=.3, size=(5, 16)).astype(np.float32)
        v = rng.normal(scale=.3, size=(50, 16)).astype(np.float32)
        u[0] = 0
        v[0] = 0
        stats = dict(known_users=np.array([False, True, True, True, True]), pop=np.r_[0, np.ones(49)],
                     ub=np.r_[0, [.2, .1, -.1, -.2]], ib=np.zeros(50), ui=np.arange(5)*10,
                     ii=np.ones(50)*20, ctr=.06)
        model = RobustMF(dict(users=u, items=v, sizes=[8, 8]), stats,
                         dict(fine_tune=fine_tune, user_dropout=0, history_event_probability=0, **kwargs))
        history = torch.zeros((3, 20), dtype=torch.long)
        history[0, :4] = torch.tensor([2, 3, 4, 5])
        age = torch.zeros((3, 20))
        age[0, :4] = torch.tensor([40., 30., 20., 10.])
        inputs = dict(users=torch.tensor([1, 0, 2]), items=torch.arange(1, 31).reshape(3, 10), history=history,
                      age_hours=age, context=torch.zeros((3, 6)), timestamps=torch.full((3,), 1737763200, dtype=torch.long))
        return model, inputs

    def test_empty_histories_permutation_and_checkpoint_reload(self):
        model, inputs = self.setup_model()
        model.eval()
        with torch.no_grad():
            p = model.probabilities(inputs)
            permutation = torch.tensor([5, 1, 8, 2, 4, 0, 7, 6, 9, 3])
            reordered = dict(inputs, items=inputs['items'][:, permutation])
            pp = model.probabilities(reordered)
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue((p.sum(1) < 1).all())
        torch.testing.assert_close(p[:, permutation], pp, atol=2e-6, rtol=2e-6)
        clone, _ = self.setup_model()
        clone.load_state_dict(model.state_dict())
        clone.eval()
        with torch.no_grad():
            torch.testing.assert_close(p, clone.probabilities(inputs), atol=0, rtol=0)

    def test_online_deletion_recomputes_context_without_mutating_inputs(self):
        model, inputs = self.setup_model()
        model.config.update(history_event_probability=1, history_delete_probability=1, user_dropout=1)
        model.train()
        before = {k:v.clone() for k,v in inputs.items()}
        features = model.raw_features(inputs)
        context_start = 17*len(model.sizes)
        self.assertFalse(features[:, :, context_start+3:context_start+6].any())
        self.assertTrue(torch.isfinite(features).all())
        for key in inputs:
            torch.testing.assert_close(inputs[key], before[key], atol=0, rtol=0)

    def test_embedding_fine_tuning_and_padding_gradients(self):
        model, inputs = self.setup_model(fine_tune=True)
        model.train()
        model(inputs).square().mean().backward()
        self.assertGreater(float(model.items.weight.grad.abs().sum()), 0)
        self.assertFalse(model.items.weight.grad[0].any())
        self.assertFalse(model.users.weight.grad[0].any())
        frozen, _ = self.setup_model(fine_tune=False)
        self.assertFalse(frozen.users.weight.requires_grad)
        self.assertFalse(frozen.items.weight.requires_grad)


if __name__ == '__main__':
    unittest.main()
