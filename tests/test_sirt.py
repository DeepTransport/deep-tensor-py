from itertools import product
import unittest

import torch
from torch.linalg import norm, solve

import deep_tensor as dt
from examples.ou_process.ou import OU


torch.manual_seed(0)


class TestSIRT(unittest.TestCase):
    """Verifies that the Rosenblatt transport and inverse Rosenblatt
    transport methods of a SIRT are actually inverses of one another 
    using a model of an OU process.
    """

    def build_ou_sirt(
        self,
        poly: dt.Basis1D, 
        method: str, 
        dim: int=20
    ) -> dt.TTSIRT:

        model = OU(d=dim, a=0.5)

        def potential_func(x: torch.Tensor):
            return model.eval_potential(x)
        
        n_samp = 1_000
        xs_samp = solve(model.B, torch.randn((dim, n_samp))).T
        input_data = dt.InputData(xs_samp)

        domain = dt.BoundedDomain(bounds=torch.tensor([-5.0, 5.0]))
        bases = dt.ApproxBases(polys=poly, domains=domain, dim=dim) 
        options = dt.TTOptions(tt_method=method, max_rank=20, max_als=1) 
        
        sirt = dt.TTSIRT(
            potential_func, 
            bases, 
            options=options, 
            input_data=input_data
        )
        
        return sirt

    def test_ou_sirt(self):

        polys = [
            dt.Legendre(order=40),
            dt.Fourier(order=20),
            dt.Lagrange1(num_elems=40),
            # dt.LagrangeP(order=5, num_elems=8)
        ]

        tt_methods = ["random", "fixed_rank"]

        dim = 20
        zs = torch.rand((10_000, dim))

        for poly, tt_method in product(polys, tt_methods):
            with self.subTest(poly=poly, tt_method=tt_method):
                
                sirt = self.build_ou_sirt(poly, tt_method, dim)
                xs = sirt.eval_irt_nograd(zs)[0]
                z0 = sirt.eval_rt(xs)
                transform_error = norm(zs-z0, ord="fro")

                self.assertTrue(transform_error < 1e-8)
        
        return