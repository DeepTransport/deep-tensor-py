import unittest

import torch
from torch import Tensor
from torch.linalg import norm

import deep_tensor as dt

from tests.ou import build_ou_sirt


torch.manual_seed(0)


class TestSIRT(unittest.TestCase):

    def build_sirts(
        self, 
        polys: list[dt.Basis1D], 
        tt_methods: list[str], 
        dim: int = 20
    ) -> dict[int, dict[int, dt.TTSIRT]]:

        sirts = {}
        for poly in polys:
            sirts[poly] = {}
            for tt_method in tt_methods:
                sirts[poly][tt_method] = build_ou_sirt(poly, tt_method, dim)

        return sirts

    def test_ou_sirt(self):
        """Verifies that the RT and IRT methods of a SIRT are actually 
        inverses of one another.
        """

        polys = [
            dt.Chebyshev1st(order=20),
            dt.Chebyshev2nd(order=20),
            dt.Fourier(order=20),
            dt.Lagrange1(num_elems=40),
            dt.LagrangeP(order=5, num_elems=8),
            dt.Legendre(order=40)
        ]

        tt_methods = ["fixed_rank", "random", "amen"]

        sirts = self.build_sirts(polys, tt_methods)

        dim = 20
        zs = torch.rand((1000, dim))

        for poly in sirts:
            for tt_method in sirts[poly]:
                with self.subTest(poly=poly, tt_method=tt_method):
                    sirt = sirts[poly][tt_method]
                    xs = sirt.eval_irt(zs)[0]
                    z0 = sirt.eval_rt(xs)
                    transform_error = norm(zs-z0, ord="fro")
                    self.assertTrue(transform_error < 1e-8)
        
        return
    
    def test_ou_sirt_marginal(self):
        """Verifies that the marginal RT and IRT are inverses of one 
        another.
        """

        polys = [
            dt.Chebyshev1st(order=20),
            dt.Chebyshev2nd(order=20),
            dt.Fourier(order=20),
            dt.Lagrange1(num_elems=40),
            dt.LagrangeP(order=5, num_elems=8),
            dt.Legendre(order=40)
        ]

        tt_methods = ["fixed_rank", "random"]
        subsets = ["first", "last"]
        dim = 20

        sirts = self.build_sirts(polys, tt_methods, dim)
        zs = torch.rand((1000, dim))

        for poly in sirts:
            for tt_method in sirts[poly]:
                for subset in subsets:
                    with self.subTest(poly=poly, tt_method=tt_method, subset=subset):

                        sirt = sirts[poly][tt_method]

                        if subset == "first":
                            indices = torch.arange(8)
                        else:
                            indices = torch.arange(dim-1, 14, -1)
                        
                        xs = sirt.eval_irt(zs[:, indices], subset)[0]
                        z0 = sirt.eval_rt(xs, subset)
                        transform_error = norm(zs[:, indices]-z0, ord="fro")
                        self.assertTrue(transform_error < 1e-8)
        
        return
    
    def compute_potential_grad_fd(
        self, 
        sirt: dt.TTSIRT, 
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

        neglogfxs_0 = sirt.eval_potential(xs_0)
        neglogfxs_1 = sirt.eval_potential(xs_1)
        grad = (neglogfxs_1 - neglogfxs_0) / (2 * dx)
        return grad.reshape(*xs.shape)

    def test_potential_grad(self):
        """Verifies that the gradient of the potential function is 
        computed correctly using a finite difference approximation.
        """

        polys = [
            dt.Chebyshev1st(order=20),
            dt.Chebyshev2nd(order=20),
            dt.Fourier(order=20),
            dt.Lagrange1(num_elems=40),
            dt.LagrangeP(order=5, num_elems=8),
            dt.Legendre(order=40)
        ]

        tt_methods = ["random"]
        grad_methods = ["manual", "autodiff"]
        dim = 20

        sirts = self.build_sirts(polys, tt_methods, dim)
        zs = torch.rand((1000, dim))

        for poly in sirts:
            for tt_method in sirts[poly]:
                for grad_method in grad_methods:
                    with self.subTest(poly=poly, tt_method=tt_method, grad_method=grad_method):

                        sirt = sirts[poly][tt_method]
                        xs = sirt.eval_rt(zs)

                        grads = sirt.eval_potential_grad(xs, method=grad_method)
                        grads_fd = self.compute_potential_grad_fd(sirt, xs)
                            
                        grad_error = norm(grads - grads_fd)
                        self.assertTrue(grad_error < 1e-4)

        return


if __name__ == "__main__":
    unittest.main()