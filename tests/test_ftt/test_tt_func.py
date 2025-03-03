import unittest

import torch
from torch import Tensor
from torch.linalg import norm

import deep_tensor as dt

from tests.ou import build_ou_sirt


torch.manual_seed(0)


class TestSIRT(unittest.TestCase):

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

        G_213 = dt.TTFunc.eval_core_213(poly, A, ls)
        G_231 = dt.TTFunc.eval_core_231(poly, A, ls)

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

        dummy_func = lambda _: 1.0

        poly = dt.Lagrange1(num_elems=2)
        domain = dt.BoundedDomain()
        dim = 3
        bases = dt.ApproxBases(poly, domain, dim)

        tt_func = dt.TTFunc(
            dummy_func, 
            bases, 
            options=dt.TTOptions(),
            input_data=dt.InputData()
        )

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

        tt_func.tt_data.cores = {
            0: A_0,
            1: A_1,
            2: A_2
        }

        ls_marg = torch.tensor([[-0.5, -0.5, -0.5],
                                [-0.5,  0.0,  0.5]])
        
        ps_forward = tt_func.eval_local(ls_marg, dt.Direction.FORWARD)
        ps_backward = tt_func.eval_local(ls_marg, dt.Direction.BACKWARD)

        ps_true = torch.tensor([[33.1250], [24.5]])

        self.assertTrue(norm(ps_forward - ps_true) < 1e-8)
        self.assertTrue(norm(ps_backward - ps_true) < 1e-8)
        return

    def test_build_block_local(self):
        """Verifies that build_block_local is working as intended.
        """

        def target_func(ls: torch.Tensor):
            return ls.sum(dim=1)
        
        poly = dt.Lagrange1(num_elems=2)
        domain = dt.BoundedDomain()
        dim = 3
        bases = dt.ApproxBases(poly, domain, dim)

        tt_func = dt.TTFunc(
            target_func, 
            bases, 
            options=dt.TTOptions(init_rank=3),
            input_data=dt.InputData()
        )

        ls_left = torch.tensor([[0.5],
                                [-0.5],
                                [1.0]])

        ls_right = torch.tensor([[0.0],
                                 [0.5],
                                 [1.0]])

        F_k = tt_func.build_block_local(ls_left, ls_right, 1)

        F_k_true = torch.tensor([[[-0.5,  0.0,  0.5],
                                  [ 0.5,  1.0,  1.5],
                                  [ 1.5,  2.0,  2.5]],
                                 [[-1.5, -1.0, -0.5],
                                  [-0.5,  0.0,  0.5],
                                  [ 0.5,  1.0,  1.5]],
                                 [[ 0.0,  0.5,  1.0],
                                  [ 1.0,  1.5,  2.0],
                                  [ 2.0,  2.5,  3.0]]])

        self.assertTrue(norm(F_k - F_k_true) < 1e-8)
        return
    
    def compute_grad_fd(
        self,
        tt_func: dt.TTFunc, 
        xs: Tensor, 
        dx: float = 1e-6
    ) -> Tensor:
        """Computes a finite difference approximation to the gradient of 
        the potential function.
        """
        n_xs, d_xs = xs.shape
        dxs = torch.tile(dx * torch.eye(d_xs), (n_xs, 1))
        xs_tiled = torch.tile(xs, (1, d_xs)).reshape(-1, d_xs)
        xs_0 = xs_tiled - dxs 
        xs_1 = xs_tiled + dxs

        neglogfxs_0 = tt_func.eval(xs_0)
        neglogfxs_1 = tt_func.eval(xs_1)
        grad = (neglogfxs_1 - neglogfxs_0) / (2 * dx)
        return grad.reshape(*xs.shape)
    
    def test_grad_ftt(self):

        poly = dt.Legendre(order=40)
        tt_method = "random"
        dim = 5
        
        sirt = build_ou_sirt(poly, tt_method, dim)
        tt_func = sirt.approx

        zs = torch.rand((1000, dim))
        xs = sirt.eval_rt(zs)

        grad_methods = ["autodiff", "manual"]

        for grad_method in grad_methods:
            with self.subTest(grad_method=grad_method):
                grads = sirt.approx.grad(xs, method=grad_method)
                grads_fd = self.compute_grad_fd(tt_func, xs)
                grad_error = norm(grads - grads_fd)
                self.assertTrue(grad_error < 1e-4)

        return


if __name__ == "__main__":
    unittest.main()