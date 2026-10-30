import unittest
from arena_push084 import candidate_families


class FamilyRegression(unittest.TestCase):
    def test_four_heads_with_two_spectral_models(self):
        families = candidate_families(['novel_wide', 'novel_cross', 'novel_spectral_wide', 'novel_spectral_cross'])
        self.assertIn([3, 4], families)
        self.assertIn([1, 2, 3, 4], families)
        self.assertTrue(all(1 <= i <= 4 for family in families for i in family))
        self.assertTrue(all(len(set(family)) == len(family) for family in families))

    def test_various_campaign_sizes_have_valid_full_family(self):
        for n in [1, 4, 5, 6, 14, 23]:
            names = ['spectral'+str(i) for i in range(n)]
            families = candidate_families(names)
            self.assertIn(list(range(1, n+1)), families)
            self.assertTrue(all(1 <= i <= n for family in families for i in family))
            self.assertEqual(len(families), len({tuple(family) for family in families}))


if __name__ == '__main__':
    unittest.main()
