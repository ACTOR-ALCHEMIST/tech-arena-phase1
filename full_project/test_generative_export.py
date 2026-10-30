"""Submission serialization must preserve opaque row IDs and row alignment."""
import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from generative_export import write_submission


class ExportTests(unittest.TestCase):
    def test_row_mapping_precision_and_atomic_output(self):
        ids=np.array([[f' Row_{i}_{j} '.encode() for j in range(10)] for i in range(2)])
        values=np.arange(20,dtype=np.float32).reshape(2,10)/20
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'submission.csv';write_submission(path,ids,values)
            with path.open(newline='') as handle:rows=list(csv.reader(handle))
            self.assertEqual(rows[0],['rowid','predicted_probability'])
            self.assertEqual([r[0] for r in rows[1:]],[r.decode() for r in ids.ravel()])
            np.testing.assert_allclose([float(r[1]) for r in rows[1:]],values.ravel(),rtol=0,atol=1e-9)
            self.assertFalse((path.parent/'submission.partial.csv').exists())

    def test_reject_invalid_values_or_missing_candidates(self):
        ids=np.full((2,10),b'rowid')
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'submission.csv'
            for bad in [np.zeros((1,10)),np.full((2,10),np.nan),np.full((2,10),1.1),np.full((2,10),-.1)]:
                with self.assertRaises(ValueError):write_submission(path,ids,bad)
                self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
