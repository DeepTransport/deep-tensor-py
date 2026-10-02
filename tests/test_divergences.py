import unittest

import torch

import deep_tensor as dt


class TestDivergences(unittest.TestCase):

    def test_divergences(self):

        # Case where P is an unnormalised verion of Q
        n = 100
        neglogps = torch.zeros(n)
        neglogqs = torch.zeros(n)
        
        f_div = dt.estimate_dhell(neglogqs, neglogps)
        self.assertTrue(torch.abs(f_div) < 1e-12)

        return


if __name__ == "__main__":
    unittest.main()