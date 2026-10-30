"""Data alignment invariants; runs without downloading any language model."""
import unittest
import numpy as np
from generative_data import class_targets,permute_candidates,restore_probabilities


class GenerativeDataTests(unittest.TestCase):
    def test_no_click_and_each_clicked_slot_have_distinct_targets(self):
        labels=np.r_[np.eye(10,dtype=np.int32),np.zeros((1,10),np.int32)]
        np.testing.assert_array_equal(class_targets(labels),np.arange(11))
        with self.assertRaises(ValueError):class_targets(np.ones((1,10)))

    def test_candidate_shuffle_preserves_labels_and_restores_row_order(self):
        labels=np.r_[np.eye(10,dtype=np.int32),np.zeros((1,10),np.int32)]
        candidates=np.arange(110*3).reshape(11,10,3)
        shuffled,target,order=permute_candidates(candidates,class_targets(labels),np.random.default_rng(4))
        for i in range(10):np.testing.assert_array_equal(shuffled[i,target[i]],candidates[i,i])
        self.assertEqual(target[-1],10)
        p=np.random.default_rng(5).random((11,10))
        np.testing.assert_array_equal(restore_probabilities(np.take_along_axis(p,order,1),order),p)

    def test_generating_inputs_is_independent_of_current_targets(self):
        from unittest.mock import patch
        from generative_data import token_arrays
        data={'history':np.zeros((2,20),np.int32),'age_hours':np.zeros((2,20),np.float32),
              'timestamps':np.array([1737849600,1737849600]),'context':np.zeros((2,6)),
              'labels':np.zeros((2,10),np.int32)}
        data['history'][1,:2]=[1,2];data['age_hours'][1,:2]=[1,48]
        behavior=((np.zeros((2,4)),np.zeros((2,10,4)),np.zeros((2,20,4))),[])
        args=(np.arange(2),np.ones((2,10,3)),'2025-01-26','2025-01-24',[],np.zeros(3),np.ones(3))
        with patch('generative_data.behavioral_arrays',return_value=behavior):
            before,_=token_arrays(data,*args)
            data['labels'][:]=1
            after,_=token_arrays(data,*args)
        for key in before:np.testing.assert_array_equal(before[key],after[key])
        np.testing.assert_array_equal(before['history'][0],0)
        np.testing.assert_array_equal(before['history'][1,2:],0)


if __name__=='__main__':unittest.main()
