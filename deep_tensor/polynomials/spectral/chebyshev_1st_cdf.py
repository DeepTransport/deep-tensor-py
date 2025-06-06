from typing import Tuple

import torch
from torch import Tensor

from .chebyshev_1st import Chebyshev1st
from .spectral_cdf import SpectralCDF


class Chebyshev1stCDF(Chebyshev1st, SpectralCDF):

    def __init__(self, poly: Chebyshev1st, **kwargs):
        Chebyshev1st.__init__(self, order=2*poly.order)
        SpectralCDF.__init__(self, **kwargs)
        return
    
    def grid_measure(self, n: int) -> Tensor:
        ls = torch.linspace(*self.domain, n)
        return ls

    def eval_int_basis(self, ls: Tensor) -> Tensor:
        
        thetas = self.l2theta(ls)[:, None]

        basis_vals = -torch.hstack((
            thetas / torch.pi, 
            ((torch.tensor(2).sqrt() / torch.pi) 
                * (thetas * self.n[1:]).sin()
                / self.n[1:])
        ))

        return basis_vals
    
    def eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        thetas = self.l2theta(ls)[:, None]
        basis_vals = self.eval_int_basis(ls)
        derivs = self.norm * torch.cos(thetas * self.n)
        ws = self.eval_measure(ls)[:, None]
        derivs = derivs * ws
        return basis_vals, derivs