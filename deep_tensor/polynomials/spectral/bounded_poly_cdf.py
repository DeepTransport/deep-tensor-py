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
        basis_vals = (torch.cos(torch.outer(thetas, self.n+1)) 
                      * (self.normalising / (self.n+1)))
        
        return basis_vals
    
    def eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        
        thetas = self.l2theta(ls)
        sin_thetas = thetas.sin()
        sin_thetas[sin_thetas.abs() < EPS] = EPS

        basis_vals = (torch.cos(torch.outer(thetas, self.n+1)) 
                      * (self.normalising / (self.n+1)))
        deriv_vals = (torch.sin(torch.outer(self.n+1, thetas)) 
                      * self.normalising / sin_thetas).T
        
        mask_lhs = torch.abs(ls + 1.0) <= EPS
        if mask_lhs.sum() > 0:
            deriv_vals[mask_lhs, :] = (self.n+1) * self.normalising * torch.pow(-1.0, self.n)
        
        mask_rhs = torch.abs(ls - 1.0) <= EPS
        if mask_rhs.sum() > 0:
            deriv_vals[mask_rhs, :] = (self.n+1) * self.normalising

        check_finite(basis_vals)
        check_finite(deriv_vals)
        return basis_vals, deriv_vals