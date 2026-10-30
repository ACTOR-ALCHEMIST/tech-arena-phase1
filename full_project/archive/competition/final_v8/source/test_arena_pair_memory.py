import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arena_pair_memory import lookup, memory
from run import timestamp


class PairMemoryTests(unittest.TestCase):
    def test_lookup_missing_and_empty(self):
        q = np.array([[0, 4, 10, 30]])
        np.testing.assert_array_equal(lookup(np.array([4, 10]), np.array([2, 7]), q), [[0, 2, 7, 0]])
        np.testing.assert_array_equal(lookup(np.array([], dtype='i8'), np.array([], dtype='i'), q), np.zeros_like(q))

    def test_future_labels_excluded_and_warmup_denominator_separate(self):
        cut = timestamp('2025-01-23')
        data = dict(users=np.ones(3, 'i'), items=np.tile(np.arange(1, 11), (3, 1)),
                    timestamps=np.array([cut-120, cut-60, cut+60]), history=np.zeros((3, 20), 'i'),
                    labels=np.zeros((3, 10), 'u1'))
        data['labels'][0, 2] = 1
        data['labels'][2, 5] = 1
        data['history'][1:, 0] = 3
        observed = dict(users=np.array([1, 1]), items=np.array([3, 3]), minutes=np.array([cut//60-1440, cut//60-2], 'i'))
        with tempfile.TemporaryDirectory() as folder, patch('arena_pair_memory.snapshot', return_value=Path(folder)), \
             patch('ctr_data.load_cache', return_value=({'n_users': 3, 'n_items': 15}, data)), \
             patch('arena_feedback.feedback', return_value=observed):
            m = memory('2025-01-23')
            keys = np.array([15+3, 15+6])
            np.testing.assert_array_equal(lookup(m['exposure_keys'], m['exposure_count'], keys), [2, 2])
            np.testing.assert_array_equal(lookup(m['exposure_keys'], m['exposure_clicks'], keys), [1, 0])
            self.assertEqual(lookup(m['click_keys'], m['click_count'], np.array([18]))[0], 2)
            result = json.loads((Path(folder)/'pair_memory_v1/result.json').read_text())
            self.assertLess(result['max_fitting_timestamp'], cut)


if __name__ == '__main__':
    unittest.main()
