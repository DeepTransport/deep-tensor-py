from typing import Tuple

import torch

from .chebyshev_2nd_unweighted import Chebyshev2ndUnweighted
from .legendre import Legendre
from .spectral_cdf import SpectralCDF
from ..constants import EPS


class BoundedPolyCDF(Chebyshev2ndUnweighted, SpectralCDF):

    def __init__(self, poly: Legendre, **kwargs):
        
        Chebyshev2ndUnweighted.__init__(self, order=2*poly.order)
        SpectralCDF.__init__(self, **kwargs)
        
        return

    def grid_measure(self, n: int) -> torch.Tensor:
        
        grid = torch.linspace(*self.domain, n)
        grid[0] = self.domain[0] - EPS
        grid[-1] = self.domain[-1] + EPS
        
        return grid
    
    def eval_int_basis(self, ls: torch.Tensor) -> torch.Tensor:
        
        thetas = self.x2theta(ls)
        basis_vals = (torch.cos(torch.outer(thetas, self.n+1)) 
                      * (self.normalising / (self.n+1)))
        
        return basis_vals
    
    def eval_int_basis_newton(
        self, 
        ls: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        thetas = self.x2theta(ls)
        basis_vals = (torch.cos(torch.outer(thetas, self.n+1)) 
                      * (self.normalising / (self.n+1)))
        deriv_vals = (torch.sin(torch.outer(self.n+1, thetas)) 
                      / (torch.sin(thetas) / self.normalising)).T
        
        mask_lhs = torch.abs(ls + 1.0) < EPS
        if torch.sum(mask_lhs) > 0:
            deriv_vals[mask_lhs, :] = (self.n+1) * torch.pow(-1.0, self.n) * self.normalising
        
        mask_rhs = torch.abs(ls - 1.0) < EPS
        if torch.sum(mask_rhs) > 0:
            deriv_vals[mask_rhs, :] = (self.n+1) * self.normalising

        return basis_vals, deriv_vals