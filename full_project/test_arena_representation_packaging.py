import tempfile
import unittest
from pathlib import Path
from arena_push084_export import representation_ignore


class RepresentationPackagingTests(unittest.TestCase):
    def test_keeps_complete_assets_and_skips_active_unrelated_fit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['spherical_long_ready','spherical_alt_active','sgns_8_ready']:(root/name).mkdir()
            (root/'spherical_long_ready/result.json').write_text('{}')
            ignored=representation_ignore(root,['spherical_long_ready','spherical_alt_active','sgns_8_ready','stats.npz','resume.pt','.build.lock','factors.tmp.npz'])
            self.assertEqual(set(ignored),{'spherical_alt_active','resume.pt','.build.lock','factors.tmp.npz'})


if __name__=='__main__':unittest.main()
