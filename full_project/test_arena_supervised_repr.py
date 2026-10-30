import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from arena_bayes_mf import BayesianMF, ANCILLARY, load_model
from arena_supervised_repr import calibration_prefix, training_windows, features_for_model


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(199)
        u, v = rng.normal(size=(4, 3)).astype('f'), rng.normal(size=(12, 3)).astype('f')
        u[0] = v[0] = 0
        anc = {k:np.ones(4 if k in ['ub', 'u_precision', 'bu_precision', 'head_users'] else 12, 'f') for k in ANCILLARY}
        anc['ub'][0] = anc['ib'][0] = 0
        self.model = BayesianMF(dict(users=u, items=v), anc, dict(anchor_scale=1., fine_items=True)).eval()
        self.raw = dict(users=np.array([0, 1, 2], np.int32), items=rng.integers(0, 12, (3, 10), dtype=np.int32),
                        history=np.array([[0]*20, [1, 2, 3]+[0]*17, [4]*20], np.int32),
                        age_hours=np.tile(np.arange(20, dtype='f'), (3, 1)), context=rng.normal(size=(3, 6)).astype('f'))

    def test_features_need_no_targets_and_match_model_probability(self):
        from run import get_batch
        from scipy.special import expit
        ix = np.arange(3)
        x, names = features_for_model(self.model, self.raw, ix, 'fresh4')
        self.assertEqual(x.shape, (30, 24))
        self.assertEqual(len(set(names)), 24)
        self.assertTrue(np.isfinite(x).all())
        with torch.inference_mode():
            expected = self.model(get_batch(self.raw, ix, 'cpu', labels=False)).softmax(1)[:, :10].numpy().ravel()
        np.testing.assert_allclose(expit(x[:, -1]), expected, atol=2e-7, rtol=2e-6)

    def test_checkpoint_and_batch_partition_parity(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.save(dict(state_dict=self.model.state_dict(), config=self.model.config), Path(folder)/'model.pt')
            rebuilt = load_model(folder, 'cpu')
            full, cols = features_for_model(rebuilt, self.raw, np.arange(3), 'fresh16')
            a, ca = features_for_model(rebuilt, self.raw, np.array([0]), 'fresh16')
            b, cb = features_for_model(rebuilt, self.raw, np.array([1, 2]), 'fresh16')
            self.assertEqual(cols, ca)
            self.assertEqual(cols, cb)
            np.testing.assert_allclose(full, np.concatenate([a, b]), rtol=1e-6, atol=1e-6)

    def test_query_targets_cannot_affect_features(self):
        ix = np.arange(3)
        a, _ = features_for_model(self.model, dict(self.raw, labels=np.zeros((3, 10))), ix, 'fresh4')
        b, _ = features_for_model(self.model, dict(self.raw, labels=np.ones((3, 10))), ix, 'fresh4')
        np.testing.assert_array_equal(a, b)

    def test_honest_calibration_and_refinement_windows(self):
        from run import timestamp
        self.assertEqual(calibration_prefix('2025-01-22'), '2025-01-21')
        self.assertEqual(calibration_prefix('2025-01-23'), '2025-01-21')
        self.assertEqual(calibration_prefix('2025-01-27'), '2025-01-24')
        t = timestamp('2025-01-26')
        raw = dict(timestamps=np.array([t-4*86400, t-3*86400, t-1, t, t+1]))
        prefix, fit, cal = training_windows(raw, '2025-01-26')
        self.assertEqual(prefix, '2025-01-23')
        np.testing.assert_array_equal(fit, [0, 1, 2])
        np.testing.assert_array_equal(cal, [1, 2])


if __name__ == '__main__':
    unittest.main()
