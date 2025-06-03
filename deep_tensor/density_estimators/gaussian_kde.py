from typing import Tuple

import torch 
from torch import Tensor

from .marginal_estimator import MarginalEstimator
from ..constants import EPS
from ..domains import BoundedDomain


class GaussianKDE(MarginalEstimator):
    r"""Gaussian kernel density estimator for a given (univariate) 
    probability density function.

    Parameters
    ----------
    TODO

    """

    def __init__(
        self, 
        xs: Tensor, 
        domain: BoundedDomain,
        n_newton: int = 100,
        n_gridpoints: int = 100
    ):

        self.xs = xs
        self.domain = domain 
        self.n_newton = n_newton

        self.n = self.xs.numel()
        self.h = self._compute_h(self.xs)
        # self.sd = torch.std(xs) * self.h
        self.sd = self.h

        self.cdf_left = self._eval_kernel_ints(self.domain.left).mean()
        self.cdf_right = self._eval_kernel_ints(self.domain.right).mean()
        self.norm = self.cdf_right - self.cdf_left  # normalising constant for pdf

        self.grid = torch.linspace(*domain.bounds, n_gridpoints)
        self.cdfs_grid = self.eval_cdf(self.grid)[0]
        return

    def eval_int_diff(self, xs: Tensor, zs_cdf: Tensor) -> Tuple[Tensor, Tensor]:
        zs, dzs = self.eval_cdf(xs)
        zs = zs - zs_cdf 
        return zs, dzs
    
    @staticmethod
    def newton_step(
        xs: Tensor,
        zs: Tensor,
        dzs: Tensor,
        x0s: Tensor,
        x1s: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Carries out a single Newton iteration."""
        dxs = -zs / dzs 
        dxs[dxs.isinf() | dxs.isnan()] = 0.0
        xs = xs + dxs 
        xs = torch.clamp(xs, x0s, x1s)
        return xs, dxs
    
    def newton(self, x0s: Tensor, x1s: Tensor, zs_cdf: Tensor) -> Tensor:
        
        # z0s = self.eval_int_diff(x0s, zs_cdf)[0]
        # z1s = self.eval_int_diff(x1s, zs_cdf)[0]

        # TODO: check initial intervals

        xs = 0.5 * (x0s + x1s)  # TODO: regula falsi step

        for i in range(self.n_newton):
            zs, dzs = self.eval_int_diff(xs, zs_cdf)
            xs, dxs = self.newton_step(xs, zs, dzs, x0s, x1s)

            if zs.abs().max() < 1e-10:  # TODO proper convergence checking
                return xs

        print("not converged...")
        return xs
    
    @staticmethod
    def _compute_h(xs: Tensor) -> Tensor:
        """Computes the bandwidth parameter using Silverman's rule."""
        n = xs.numel()
        iqr = torch.quantile(xs, 0.75) - torch.quantile(xs, 0.25)
        sd = torch.std(xs)
        h = 0.9 * torch.min(sd, iqr/1.34) * n ** -0.2
        return h
    
    def _eval_kernels(self, xs: Tensor) -> Tensor:
        """Evaluates each kernel at each value of xs."""
        xs = torch.atleast_1d(xs)
        dxs = xs[:, None] - self.xs
        kernels = ((1.0 / (self.sd * torch.tensor(2.0*torch.pi).sqrt())) 
                   * torch.exp(-dxs.square() / (2 * self.sd.square())))
        return kernels
    
    def _eval_kernel_ints(self, xs: Tensor) -> Tensor:
        """Evaluates the integral of each kernel at each value of xs."""
        xs = torch.atleast_1d(xs)
        dxs = xs[:, None] - self.xs 
        kernel_ints = 0.5 * (1.0 + torch.erf(dxs / (self.sd * torch.tensor(2.0).sqrt())))
        return kernel_ints

    def eval_pdf(self, xs: Tensor) -> Tensor:
        kernels = self._eval_kernels(xs)
        pdfs = kernels.mean(dim=1) / self.norm
        return pdfs
    
    def eval_potential(self, xs: Tensor) -> Tensor:
        return -self.eval_pdf(xs).log()
    
    def eval_cdf(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        kernel_ints = self._eval_kernel_ints(xs)
        cdfs = (kernel_ints.mean(dim=1) - self.cdf_left) / self.norm
        pdfs = self.eval_pdf(xs)
        return cdfs, pdfs
    
    def invert_cdf(self, zs: Tensor) -> Tensor:

        zs = zs.clamp(EPS, 1.0-EPS)

        inds_left = ((self.cdfs_grid[:, None] - zs) >= 0).int().argmax(dim=0) - 1
        inds_left = inds_left.clamp(0, self.n-2)

        x0s = self.grid[inds_left]
        x1s = self.grid[inds_left+1]
        xs = self.newton(x0s, x1s, zs)
        return xs