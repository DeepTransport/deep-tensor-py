from typing import Tuple

import torch

from .spectral_cdf import SpectralCDF
from ..polynomials.chebyshev_1st import Chebyshev1st
from ...constants import EPS


class Chebyshev1stCDF(Chebyshev1st, SpectralCDF):

    def __init__(self, poly: Chebyshev1st, **kwargs):
        Chebyshev1st.__init__(self, order=2*poly.order)
        SpectralCDF.__init__(self, **kwargs)
        return
    
    def grid_measure(self, n: int) -> torch.Tensor:
        x = torch.linspace(*self.domain, n)
        x[0] = self.domain[0] - EPS
        x[-1] = self.domain[1] + EPS
        return x

    def eval_int_basis(self, xs: torch.Tensor) -> torch.Tensor:
        
        thetas = self.l2theta(xs)

        if self.order == 0:
            basis_vals = -(thetas / torch.pi).reshape(-1, 1)
            return basis_vals

        basis_vals = -torch.hstack((
            (thetas / torch.pi).reshape(-1, 1), 
            ((torch.tensor(2).sqrt() / torch.pi) 
                * torch.sin(torch.outer(thetas, self.n[1:])) 
                / self.n[1:])
        ))

        return basis_vals
    
    def eval_int_basis_newton(
        self, 
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        basis_vals = self.eval_int_basis(xs)

        theta = self.l2theta(xs)
        derivs = torch.cos(torch.outer(theta, self.n)) * self.normalising
        w = self.eval_measure(xs)
        derivs = derivs * w[:, None]
        
        return basis_vals, derivs