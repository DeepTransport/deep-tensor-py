import unittest

import torch
from torch import Tensor
from torch.linalg import norm

import deep_tensor as dt


torch.manual_seed(0)


def dummy_pdf(xs: Tensor):
    return (xs.square() + 1.0).log() * (-0.5 * xs.square()).exp()


class TestCDF(unittest.TestCase):
    
    def test_eval_invert(self):
        """Verifies that the eval() and invert() methods of each CDF 
        class are inverses of one another.
        """

        n_ls = 10_000

        polys = {
            "Chebyshev1st": dt.Chebyshev1st(order=20),
            "Chebyshev2nd": dt.Chebyshev2nd(order=20),
            "Fourier": dt.Fourier(order=20),
            "Lagrange1": dt.Lagrange1(num_elems=20),
            "LagrangeP": dt.LagrangeP(order=5, num_elems=8),
            "Legendre": dt.Legendre(order=20)
        }

        for poly in polys:
            with self.subTest(poly=poly):
            
                cdf = dt.construct_cdf(polys[poly])

                ls = torch.linspace(-1.0, 1.0, n_ls)
                ps = dummy_pdf(cdf.nodes) + 1e-2
                ps = ps.tile(n_ls, 1).T

                zs = cdf.eval_cdf(ps, ls)
                ls_0 = cdf.invert_cdf(ps, zs)
                
                self.assertTrue(norm(ls-ls_0) < 1e-8)

        return


if __name__ == "__main__":
    unittest.main()