import math
import unittest

import torch
from torch import Tensor

import deep_tensor as dt

torch.manual_seed(0)


class TestDIRTStandardGaussian(unittest.TestCase):
    """Some tests with the standard multivariate Gaussian density.
    """

    @staticmethod
    def neglogpri(xs: Tensor) -> Tensor:
        return 0.5 * xs.square().sum(dim=1)

    @staticmethod
    def negloglik(xs: Tensor) -> Tensor:
        return torch.zeros((xs.shape[0],)) 

    def build_dirt(
        self, 
        dim: int = 5, 
        bases: dt.Basis1D | None = None
    ):
        
        if bases is None:
            bases = dt.Lagrange1(num_elems=30)

        preconditioner = dt.IdentityMapping(dim=dim)

        # bounds = torch.tensor([[-4.0] * dim, [4.0] * dim]).T
        # preconditioner = dt.UniformMapping(bounds, reference=dt.UniformReference())

        tt_options = dt.TTOptions(verbose=0)
        dirt_options = dt.DIRTOptions(verbose=False)

        dirt = dt.DIRT(
            self.negloglik, 
            self.neglogpri, 
            preconditioner=preconditioner, 
            bases=bases,
            tt_options=tt_options,
            dirt_options=dirt_options
        )
        
        return dirt
    
    def _test_sampling(self, dirt: dt.DIRT):
        """Verifies whether the eval_irt method is working as intended.
        """

        n_samples = 50_000
        rs = dirt.reference.random(dirt.dim, n_samples)

        xs, potentials_dirt = dirt.eval_irt(rs)
        mean_dirt = xs.mean(dim=0)
        cov_dirt = xs.T.cov()

        mean_true = torch.zeros((dirt.dim,))
        cov_true = torch.eye(dirt.dim)
        neglognorm = 0.5 * dirt.dim * math.log(2.0 * math.pi)
        potentials_true = neglognorm + self.neglogpri(xs)

        self.assertTrue((mean_dirt-mean_true).abs().max() < 0.1)
        self.assertTrue((cov_dirt-cov_true).abs().max() < 0.1)
        self.assertTrue((potentials_true-potentials_dirt).abs().mean() < 0.1)

        # from plotting import plot_potentials
        # plot_potentials(potentials_dirt, potentials_true)

        return
    
    def _test_rt_irt(self, dirt: dt.DIRT):
        """Verifies whether the Rosenblatt transport and inverse 
        Rosenblatt transport methods are inverses of one another.
        """

        n_samples = 10_000
        rs = dirt.reference.random(dirt.dim, n_samples)

        xs_dirt = dirt.eval_irt(rs)[0]
        rs_dirt = dirt.eval_rt(xs_dirt)[0]

        self.assertTrue((rs-rs_dirt).abs().max() < 1e-2)
        return

    def test_dimensions(self):

        dims = [5, 10, 20]

        for dim in dims:
            with self.subTest(dim=dim):
                dirt = self.build_dirt(dim=dim)
                self._test_sampling(dirt)

        return
    
    def test_bases(self):
        """Tests the sampling and Rosenblatt transport methods when 
        different approximation bases are used.
        """

        bases_list = [
            dt.Chebyshev1st(order=30),
            dt.Chebyshev2nd(order=30),
            dt.Lagrange1(num_elems=30),
            dt.LagrangeP(order=5, num_elems=4),
            dt.Legendre(order=30),
            dt.Fourier(order=15)
        ]

        for bases in bases_list:
            with self.subTest(bases=bases):
                dirt = self.build_dirt(bases=bases)
                self._test_sampling(dirt)
                self._test_rt_irt(dirt)

        return


if __name__ == "__main__":
    unittest.main()