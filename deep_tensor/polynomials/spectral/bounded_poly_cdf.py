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
        """Evaluates the integral of each basis function at each 
        element in ls.
        """
        thetas = self.l2theta(ls)
        thetas = thetas[:, None]
        int_ps = (thetas * (self.n+1)).cos() * self.norm / (self.n+1)
        check_finite(int_ps)
        return int_ps
    
    def eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        int_ps = self.eval_int_basis(ls)
        ps = self.eval_basis(ls)
        return int_ps, ps