import unittest
import numpy as np
from arena_novel_choice import examples


class ChoiceTest(unittest.TestCase):
    def test_fit_targets_past_and_novel_only(self):
        data = dict(timestamps=np.array([1, 2, 3, 10]), labels=np.array([[1, 0], [0, 1], [0, 0], [1, 0]]),
                    items=np.array([[4, 5], [4, 5], [4, 5], [4, 5]]), history=np.array([[0, 0], [5, 0], [0, 0], [0, 0]]))
        from unittest.mock import patch
        with patch('run.timestamp', return_value=5):
            indices, allowed, targets = examples(data, 'cutoff')
        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_array_equal(allowed, [[True, True]])
        np.testing.assert_array_equal(targets, [0])


if __name__ == '__main__':
    unittest.main()
