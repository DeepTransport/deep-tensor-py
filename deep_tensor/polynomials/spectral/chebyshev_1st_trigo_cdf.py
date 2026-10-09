import math
from typing import Tuple

import torch
from torch import Tensor

from .chebyshev_1st import Chebyshev1st
from .trigo_cdf import TrigoCDF


class Chebyshev1stTrigoCDF(TrigoCDF, Chebyshev1st):

    def __init__(self, poly: Chebyshev1st, error_tol: float):
        order = 2 * poly._order
        Chebyshev1st.__init__(self, order=order, device=poly._device)
        TrigoCDF.__init__(self, error_tol=error_tol, device=poly._device)
        return

    @property
    def _domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0], device=self._device)

    @property
    def _cardinality(self) -> int:
        return self._nodes.numel()

    def _eval_int_basis(self, thetas: Tensor) -> Tensor:
        thetas = thetas[:, None]
        # Cui et al, 2023
        int_pws = torch.hstack((
            thetas / torch.pi,
            math.sqrt(2.0) / (torch.pi * self._n[1:])
                * torch.sin(thetas * self._n[1:]),
        ))
        return int_pws

    def _eval_int_basis_newton(self, thetas: Tensor) -> Tuple[Tensor, Tensor]:
        int_pws = self._eval_int_basis(thetas)
        thetas = thetas[:, None]
        derivs = self._norm * torch.cos(thetas * self._n) / torch.pi
        return int_pws, derivs