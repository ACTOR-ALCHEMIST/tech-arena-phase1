import unittest
import numpy as np
import torch
from arena_push_model import build_model, temporal_columns


class MaskTests(unittest.TestCase):
    def test_corrupt_dropped_features_cannot_change_predictions(self):
        torch.set_num_threads(2)
        cfg = dict(architecture='v2_wide', hidden=[32, 16], context_width=8,
                   loss='event_ce', dropout=0, input_mask_columns=[1, 3])
        mean, scale = np.arange(5, dtype='f'), np.ones(5, 'f')
        model = build_model(5, mean, scale, cfg).eval()
        x = torch.randn(7, 10, 5)
        changed = x.clone()
        changed[:, :, [1, 3]] = 1000*torch.randn(7, 10, 2)
        torch.testing.assert_close(model(x), model(changed), rtol=0, atol=0)
        restored = build_model(5, mean, scale, cfg).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(changed), restored(x), rtol=0, atol=0)

    def test_optional_training_dropout_preserves_clean_inference(self):
        cfg = dict(architecture='v2_wide', hidden=[16], context_width=0, loss='event_ce', dropout=0,
                   input_mask_columns=[1], input_mask_probability=.2, input_mask_inference=False)
        masked = build_model(3, np.zeros(3, 'f'), np.ones(3, 'f'), cfg).eval()
        plain_cfg = {k:v for k,v in cfg.items() if not k.startswith('input_mask')}
        plain = build_model(3, np.zeros(3, 'f'), np.ones(3, 'f'), plain_cfg).eval()
        plain.load_state_dict(masked.state_dict())
        x = torch.randn(4, 10, 3)
        torch.testing.assert_close(masked(x), plain(x), rtol=0, atol=0)

    def test_group_selection_is_semantic(self):
        names = ['sgns8_s42_hist_same_bucket_cos', 'other', 'user_bucket_affinity', 'sgns16_s42_hist_same_weekend_cos']
        self.assertEqual(temporal_columns(names, 'semantic'), [0, 3])
        self.assertEqual(temporal_columns(names, 'affinity'), [2])
        self.assertEqual(temporal_columns(names, 'all'), [0, 2, 3])


if __name__ == '__main__':
    unittest.main()
