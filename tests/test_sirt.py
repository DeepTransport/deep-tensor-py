from itertools import product
import unittest

import torch
from torch.linalg import norm

import deep_tensor as dt

from tests.ou import build_ou_sirt


torch.manual_seed(0)


class TestSIRT(unittest.TestCase):
    """Verifies that the Rosenblatt transport and inverse Rosenblatt
    transport methods of a SIRT are actually inverses of one another 
    using a model of an OU process.
    """

    def test_ou_sirt(self):

        polys = [
            dt.Legendre(order=40),
            dt.Fourier(order=20),
            dt.Lagrange1(num_elems=40)
        ]

        tt_methods = ["random", "fixed_rank"]

        dim = 20
        zs = torch.rand((10_000, dim))

        for poly, tt_method in product(polys, tt_methods):
            with self.subTest(poly=poly, tt_method=tt_method):
                
                sirt = build_ou_sirt(poly, tt_method, dim)
                xs = sirt.eval_irt_nograd(zs)[0]
                z0 = sirt.eval_rt(xs)
                transform_error = norm(zs-z0, ord="fro")

                self.assertTrue(transform_error < 1e-8)
        
        return
    
    def test_ou_sirt_marginal(self):

        polys = [
            dt.Legendre(order=40),
            dt.Fourier(order=20),
            dt.Lagrange1(num_elems=40)
        ]

        tt_methods = ["random", "fixed_rank"]
        directions = ["forward, backward"]

        dim = 20
        zs = torch.rand((10_000, dim))

        for poly, tt_method, direction in product(polys, tt_methods, directions):
            with self.subTest(poly=poly, tt_method=tt_method, direction=direction):

                sirt = build_ou_sirt(poly, tt_method, dim)

                if direction == "forward":
                    indices = torch.arange(8)
                    if sirt.int_dir != dt.Direction.FORWARD:
                        sirt.marginalise(dt.Direction.FORWARD) 
                else:
                    indices = torch.arange(dim-1, 14, -1)
                    if sirt.int_dir != dt.Direction.BACKWARD:
                        sirt.marginalise(dt.Direction.BACKWARD)
                
                xs = sirt.eval_irt_nograd(zs[:, indices])[0]
                z0 = sirt.eval_rt(xs)
                transform_error = norm(zs[:, indices]-z0, ord="fro")

                self.assertTrue(transform_error < 1e-8)
        
        return