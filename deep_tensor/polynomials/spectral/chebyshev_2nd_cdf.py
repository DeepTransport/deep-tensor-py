from typing import Tuple

import torch
from torch import Tensor

from .chebyshev_2nd import Chebyshev2nd
from .legendre import Legendre
from .spectral_cdf import SpectralCDF
from ...tools import check_finite


class Chebyshev2ndCDF(Chebyshev2nd, SpectralCDF):

    def __init__(self, poly: Legendre, error_tol: float):        
        Chebyshev2nd.__init__(self, order=2*poly._order, device=poly._device)
        SpectralCDF.__init__(self, error_tol, poly._device)
        return

    def _grid_measure(self, n: int) -> Tensor:
        return torch.linspace(self._domain[0], self._domain[1], n, device=self._device)
    
    def _eval_int_basis(self, ls: Tensor) -> Tensor:
        """Evaluates the integral of each basis function at each 
        element in ls.
        """
        thetas = self._l2theta(ls)[:, None]
        int_ps = (thetas * (self._n+1)).cos() * self._norm / (self._n+1)
        check_finite(int_ps)
        return int_ps
    
    def _eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        int_ps = self._eval_int_basis(ls)
        ps = self._eval_basis(ls)
        return int_ps, ps