import unittest

import torch

import deep_tensor as dt


DIVERGENCES = ("h2", "kl", "tv")


class TestDivergences(unittest.TestCase):

    def test_divergences(self):

        # Case where P is an unnormalised verion of Q
        n = 100
        logps = torch.zeros(n)
        logqs = torch.zeros(n)
        
        for div in DIVERGENCES:
            with self.subTest(div=div):
                f_div = dt.compute_f_divergence(logqs, logps, div)
                self.assertTrue(torch.abs(f_div) < 1e-12)

        return


if __name__ == "__main__":
    unittest.main()