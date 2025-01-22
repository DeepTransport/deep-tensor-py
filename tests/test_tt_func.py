import unittest

import torch

import deep_tensor as dt


torch.manual_seed(0)


class TestSIRT(unittest.TestCase):

    def test_eval_core(self):
        """Verifies that the eval_oned_core_213 and eval_oned_core_231 
        methods are working as intended.
        """

        poly = dt.Lagrange1(num_elems=2)

        A = torch.tensor([[[1.0, 2.0], 
                           [3.0, 1.0], 
                           [1.0, 4.0]], 
                          [[3.0, 2.0], 
                           [1.0, 2.0],
                           [2.0, 3.0]]])

        ls = torch.tensor([-0.5, 0.0, 0.5])

        G_213 = dt.TTFunc.eval_oned_core_213(poly, A, ls)
        G_231 = dt.TTFunc.eval_oned_core_231(poly, A, ls)

        G_213_true = torch.tensor([[2.0, 1.5],
                                   [2.0, 2.0],
                                   [3.0, 1.0],
                                   [1.0, 2.0],
                                   [2.0, 2.5],
                                   [1.5, 2.5]])
    
        G_231_true = torch.tensor([[2.0, 2.0],
                                   [1.5, 2.0],
                                   [3.0, 1.0],
                                   [1.0, 2.0],
                                   [2.0, 1.5],
                                   [2.5, 2.5]])

        self.assertTrue(G_213.shape == torch.Size([6, 2]))
        self.assertTrue(G_231.shape == torch.Size([6, 2]))
        self.assertTrue((G_213-G_213_true).max().abs() < 1e-8)
        self.assertTrue((G_231-G_231_true).max().abs() < 1e-8)
        return