import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from arena_bayes_mf import BayesianMF, ANCILLARY, load_model, fit_indices, anchor


class BayesianTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(42)
        u, v = rng.normal(size=(4, 2)).astype('f'), rng.normal(size=(12, 2)).astype('f')
        u[0] = v[0] = 0
        anc = {k:np.ones(4 if k.startswith(('u_', 'bu_', 'head_u')) or k == 'ub' else 12, 'f') for k in ANCILLARY}
        anc['head_users'] = np.array([1, 3, 1, 1], 'f')
        for k in ['u_precision', 'v_precision', 'bu_precision', 'bi_precision', 'ub', 'ib']:
            anc[k][0] = 0
        self.model = BayesianMF(dict(users=u, items=v), anc, dict(anchor_scale=1., fine_items=True))
        self.batch = dict(users=torch.tensor([1, 1, 1, 2]), items=torch.tensor(np.tile(np.arange(1, 11), (4, 1))),
                          history=torch.tensor(np.tile([1, 2, 3]+[0]*17, (4, 1))),
                          age_hours=torch.arange(20).float().expand(4, -1), context=torch.zeros(4, 6))

    def test_prior_is_applied_once_per_id(self):
        with torch.no_grad():
            self.model.users.weight[1:] += .1
        expected = .5*(2*.01+2*.01)/4
        self.assertAlmostEqual(self.model.penalty(self.batch).item(), expected, places=7)

    def test_candidate_equivariance_and_checkpoint(self):
        self.model.configure(True)
        expected = self.model(self.batch)
        perm = torch.tensor([9, 0, 8, 1, 7, 2, 6, 3, 5, 4])
        changed = self.model(dict(self.batch, items=self.batch['items'][:, perm]))
        torch.testing.assert_close(changed[:, :10], expected[:, perm])
        torch.testing.assert_close(changed[:, 10], expected[:, 10])
        with tempfile.TemporaryDirectory() as folder:
            torch.save(dict(state_dict=self.model.state_dict(), config=self.model.config), Path(folder)/'model.pt')
            torch.testing.assert_close(load_model(folder, 'cpu')(self.batch), expected)

    def test_item_updates_do_not_change_fixed_history_covariates(self):
        before = self.model.features(self.batch)
        with torch.no_grad():
            self.model.items.weight[1:] += .2
        after = self.model.features(self.batch)
        torch.testing.assert_close(before[:, :, 1:], after[:, :, 1:])
        self.assertGreater(float((before[:, :, 0]-after[:, :, 0]).abs().max()), .01)

    def test_empty_history_cold_id_gradients(self):
        self.model.configure(True)
        b = dict(self.batch, users=torch.zeros(4, dtype=torch.long), history=torch.zeros(4, 20, dtype=torch.long))
        b['items'] = b['items'].clone()
        b['items'][:, 0] = 0
        loss = self.model(b).square().mean()+self.model.penalty(b)
        loss.backward()
        for p in self.model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue((self.model.users.weight.grad[0] == 0).all())
        self.assertTrue((self.model.items.weight.grad[0] == 0).all())

    def test_scale_freezes_during_posterior_updates(self):
        self.model.config['fine_items'] = False
        self.model.configure(False)
        self.assertTrue(self.model.coefficients.requires_grad)
        self.assertFalse(self.model.users.weight.requires_grad)
        self.model.configure(True)
        self.assertFalse(self.model.coefficients.requires_grad)
        self.assertFalse(self.model.outside.requires_grad)
        self.assertTrue(self.model.users.weight.requires_grad)
        self.assertFalse(self.model.items.weight.requires_grad)

    def test_training_window_excludes_future_labels(self):
        from run import timestamp
        t = timestamp('2025-01-26')
        raw = dict(timestamps=np.array([t-3*86400-1, t-3*86400, t-1, t, t+1]))
        np.testing.assert_array_equal(fit_indices(raw, '2025-01-26'), [1, 2])
        np.testing.assert_array_equal(fit_indices(raw, '2025-01-26', full=True), [0, 1, 2])
        self.assertEqual(anchor('2025-01-24'), '2025-01-21')
        self.assertEqual(anchor('2025-01-27'), '2025-01-23')


if __name__ == '__main__':
    unittest.main()
