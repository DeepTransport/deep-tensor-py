from typing import Tuple

import torch
from torch import Tensor

from .chebyshev_2nd_unweighted import Chebyshev2ndUnweighted
from .legendre import Legendre
from .spectral_cdf import SpectralCDF
from ...constants import EPS
from ...tools import check_finite


class BoundedPolyCDF(Chebyshev2ndUnweighted, SpectralCDF):

    def __init__(self, poly: Legendre, **kwargs):        
        Chebyshev2ndUnweighted.__init__(self, order=2*poly.order)
        SpectralCDF.__init__(self, **kwargs)
        return

    def grid_measure(self, n: int) -> Tensor:
        grid = torch.linspace(*self.domain, n)
        grid = torch.clamp(grid, self.domain[0]+EPS, self.domain[-1]-EPS)
        return grid
    
    def eval_int_basis(self, ls: Tensor) -> Tensor:
        thetas = self.l2theta(ls)
        thetas = thetas[:, None]
        ps = (thetas * (self.n+1)).cos() * self.norm / (self.n+1)
        return ps
    
    def eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        
        thetas = self.l2theta(ls)
        thetas = thetas[:, None]
        sin_thetas = thetas.sin()
        sin_thetas[sin_thetas.abs() < EPS] = EPS

        ps = (thetas * (self.n+1)).cos() * self.norm / (self.n+1)
        dpdls = (thetas * (self.n+1)).sin() * self.norm / sin_thetas
        
        if (mask_lhs := (ls + 1.0).abs() <= EPS).sum() > 0:
            dpdls[mask_lhs, :] = (self.n+1) * self.norm * torch.pow(-1.0, self.n)
        
        if (mask_rhs := (ls - 1.0).abs() <= EPS).sum() > 0:
            dpdls[mask_rhs, :] = (self.n+1) * self.norm

        check_finite(ps)
        check_finite(dpdls)
        return ps, dpdls