import math
from typing import Tuple

import torch
from torch import Tensor

from .chebyshev_1st import Chebyshev1st
from .spectral_cdf import SpectralCDF


class Chebyshev1stCDF(Chebyshev1st, SpectralCDF):

    def __init__(self, poly: Chebyshev1st, error_tol: float):
        order = 2*poly._order
        Chebyshev1st.__init__(self, order=order, device=poly._device)
        SpectralCDF.__init__(self, error_tol=error_tol, device=poly._device)
        return
    
    def _grid_measure(self, n: int) -> Tensor:
        ls = torch.linspace(-1.0, 1.0, n, device=self._device)
        return ls

    def _eval_int_basis(self, ls: Tensor) -> Tensor:
        thetas = self._l2theta(ls)[:, None]
        basis_vals = -torch.hstack((
            thetas / torch.pi, 
            ((math.sqrt(2.0) / torch.pi) 
                * torch.sin(thetas * self._n[1:]) / self._n[1:])
        ))
        return basis_vals
    
    def _eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        thetas = self._l2theta(ls)[:, None]
        basis_vals = self._eval_int_basis(ls)
        derivs = self._norm * torch.cos(thetas * self._n)
        ws = self._eval_measure(ls)[:, None]
        derivs = derivs * ws
        return basis_vals, derivs