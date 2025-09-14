import unittest

import torch
from torch.linalg import norm

import deep_tensor as dt


torch.manual_seed(0)


class TestTTFunc(unittest.TestCase):

    def test_eval_core(self):
        """Verifies that the eval_core_213 and eval_core_231 
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

        G_213 = dt.FTT.eval_core(poly, A, ls)
        G_231 = dt.FTT.eval_core_rev(poly, A, ls)

        G_213_true = torch.tensor([[[2.0, 1.5],
                                    [2.0, 2.0]],
                                   [[3.0, 1.0],
                                    [1.0, 2.0]],
                                   [[2.0, 2.5],
                                    [1.5, 2.5]]])
    
        G_231_true = torch.tensor([[[2.0, 2.0],
                                    [1.5, 2.0]],
                                   [[3.0, 1.0],
                                    [1.0, 2.0]],
                                   [[2.0, 1.5],
                                    [2.5, 2.5]]])

        self.assertTrue(G_213.shape == torch.Size([3, 2, 2]))
        self.assertTrue(G_231.shape == torch.Size([3, 2, 2]))
        self.assertTrue((G_213-G_213_true).max().abs() < 1e-8)
        self.assertTrue((G_231-G_231_true).max().abs() < 1e-8)
        return
    
    def test_eval_local(self):
        """Verifies that eval_local is working as intended (when 
        evaluating the marginal PDF).
        """

        dim = 3
        
        basis = dt.Lagrange1(num_elems=2)
        bases = dt.ApproxBases(basis, dim)
        tt_options = dt.TTOptions(verbose=0)
        tt = dt.TT(tt_options)
        ftt = dt.FTT(bases, tt)

        A_0 = torch.tensor([[[1.0, 2.0], 
                             [2.0, 2.0], 
                             [1.0, 3.0]]])
        A_1 = torch.tensor([[[1.0, 2.0], 
                             [3.0, 1.0], 
                             [1.0, 4.0]], 
                            [[3.0, 2.0], 
                             [1.0, 2.0],
                             [2.0, 3.0]]])
        A_2 = torch.tensor([[[2.0], 
                             [3.0], 
                             [2.0]], 
                            [[4.0], 
                             [1.0], 
                             [2.0]]])

        ftt.tt.cores = {
            0: A_0,
            1: A_1,
            2: A_2
        }

        ftt.compute_cores()

        ls_marg = torch.tensor([[-0.5, -0.5, -0.5],
                                [-0.5,  0.0,  0.5]])
        
        ps_forward = ftt.eval(ls_marg, dt.Direction.FORWARD)
        ps_backward = ftt.eval(ls_marg, dt.Direction.BACKWARD)

        ps_true = torch.tensor([[33.1250], [24.5]])

        self.assertTrue(norm(ps_forward - ps_true) < 1e-8)
        self.assertTrue(norm(ps_backward - ps_true) < 1e-8)
        return

    # def test_build_block_local(self):
    #     """Verifies that build_block_local is working as intended.
    #     """

    #     def target_func(ls: torch.Tensor):
    #         return ls.sum(dim=1)
        
    #     poly = dt.Lagrange1(num_elems=2)
    #     domain = dt.BoundedDomain()
    #     dim = 3
    #     bases = dt.ApproxBases(poly, domain, dim)
    #     options = dt.TTOptions()
    #     input_data = dt.InputData()
    #     reference = dt.GaussianReference()

    #     tt_func = dt.FTT(
    #         target_func, 
    #         bases, 
    #         options=options,
    #         input_data=input_data,
    #         reference=reference
    #     )


    #     ls_left = torch.tensor([[0.5],
    #                             [-0.5],
    #                             [1.0]])

    #     ls_right = torch.tensor([[0.0],
    #                              [0.5],
    #                              [1.0]])

    #     F_k = tt_func.tt.compute_block(ls_left, ls_right, 1)

    #     F_k_true = torch.tensor([[[-0.5,  0.0,  0.5],
    #                               [ 0.5,  1.0,  1.5],
    #                               [ 1.5,  2.0,  2.5]],
    #                              [[-1.5, -1.0, -0.5],
    #                               [-0.5,  0.0,  0.5],
    #                               [ 0.5,  1.0,  1.5]],
    #                              [[ 0.0,  0.5,  1.0],
    #                               [ 1.0,  1.5,  2.0],
    #                               [ 2.0,  2.5,  3.0]]])

    #     self.assertTrue(norm(F_k - F_k_true) < 1e-8)
    #     return


if __name__ == "__main__":
    unittest.main()