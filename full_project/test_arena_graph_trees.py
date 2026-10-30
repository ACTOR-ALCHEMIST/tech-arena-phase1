import tempfile
import unittest
from pathlib import Path
import numpy as np
from arena_graph_trees import matrix, predict, CatBoostClassifier


class TreeTests(unittest.TestCase):
    def test_neural_offset_checkpoint_composition(self):
        import lightgbm as lgb
        import torch
        from arena_graph_trees import neural_logit
        from arena_graph_lgbm import params, fit_binned
        from arena_push_model import build_model
        from scipy.special import expit
        rng = np.random.default_rng(15)
        x = rng.normal(size=(1200, 4)).astype('f')
        labels = (x[:, 0]+rng.normal(size=1200) > .5).astype(int)
        cfg = dict(loss='event_ce', architecture='v2_wide', hidden=[8], context_width=0, dropout=0)
        base = build_model(4, np.zeros(4, 'f'), np.ones(4, 'f'), cfg)
        config = dict(leaves=7, neural_offset=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            torch.save(dict(state_dict=base.state_dict(), width=4, config=cfg), path/'base_model.pt')
            offset = neural_logit(path/'base_model.pt', x)
            data = lgb.Dataset(x, label=labels, init_score=offset, params=params(config))
            model = fit_binned(config, data, fixed_epochs=8)
            model.save_model(str(path/'model.txt'))
            np.testing.assert_allclose(predict(path, x).ravel(), expit(offset+model.predict(x, raw_score=True)), rtol=0, atol=1e-15)

    def test_lightgbm_text_checkpoint_replay(self):
        import lightgbm as lgb
        from arena_graph_lgbm import params, fit_binned
        rng = np.random.default_rng(72)
        x = rng.normal(size=(1200, 4)).astype('f')
        y = (x[:, 0]+rng.normal(size=1200) > 1).astype(int)
        config = dict(leaves=7)
        data = lgb.Dataset(x, label=y, params=params(config))
        model = fit_binned(config, data, fixed_epochs=8)
        with tempfile.TemporaryDirectory() as folder:
            model.save_model(str(Path(folder)/'model.txt'))
            np.testing.assert_allclose(predict(folder, x).ravel(), model.predict(x), rtol=0, atol=1e-15)

    def test_slate_permutation_and_checkpoint_replay(self):
        rng = np.random.default_rng(44)
        x = rng.normal(size=(120, 10, 4)).astype('f')
        names = ['repeat_any', 'user_bias', 'cf_cos_sum', 'other']
        values, cols = matrix(x.reshape(-1, 4), names, True)
        perm = rng.permutation(10)
        changed, changed_cols = matrix(x[:, perm].reshape(-1, 4), names, True)
        self.assertEqual(cols, changed_cols)
        np.testing.assert_allclose(changed.reshape(120, 10, -1), values.reshape(120, 10, -1)[:, perm], rtol=1e-5, atol=1e-6)
        labels = (rng.random(1200) < .1).astype(int)
        model = CatBoostClassifier(iterations=8, depth=3, verbose=False, allow_writing_files=False, thread_count=2)
        model.fit(values, labels)
        with tempfile.TemporaryDirectory() as folder:
            model.save_model(str(Path(folder)/'model.cbm'))
            np.testing.assert_array_equal(predict(folder, np.asfortranarray(values)).ravel(), model.predict_proba(values)[:, 1])


if __name__ == '__main__':
    unittest.main()
