import unittest

from score_pairs import select_fixed


class SelectionTests(unittest.TestCase):
    def test_only_fixed_margins_are_candidates(self):
        results = [
            dict(adapter="base", margin="base", gamma=None, accuracy=0.9, logloss=0.1),
            dict(adapter="fixed-g0", margin="fixed", gamma=0.0, accuracy=0.55, logloss=0.70),
            dict(adapter="fixed-g1", margin="fixed", gamma=1.0, accuracy=0.56, logloss=0.71),
            dict(adapter="fixed-g2", margin="fixed", gamma=2.0, accuracy=0.56, logloss=0.69),
            dict(adapter="calibrated", margin="calibrated", gamma=None, accuracy=0.61, logloss=0.6),
        ]
        self.assertEqual(select_fixed(results), "fixed-g2")   # accuracy tie broken by log-loss

    def test_gamma_breaks_full_ties(self):
        results = [dict(adapter=f"fixed-g{g}", margin="fixed", gamma=g, accuracy=0.5, logloss=0.7) for g in (1.0, 0.25)]
        self.assertEqual(select_fixed(results), "fixed-g0.25")


if __name__ == "__main__":
    unittest.main()
