from typing import Tuple

import torch
from torch import Tensor

from .spectral_cdf import SpectralCDF
from .fourier import Fourier


class FourierCDF(SpectralCDF):

    def __init__(self, poly: Fourier, error_tol: float):
        order = 2 * poly._order
        self._basis = Fourier(order, device=poly._device)
        self._nodes = self._basis._nodes
        self._node2basis = self._basis._node2basis
        self._m = self._basis._m 
        self._c = self._basis._c
        SpectralCDF.__init__(self, error_tol=error_tol, device=poly._device)
        return
    
    @property
    def _domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0], device=self._device)
    
    @property 
    def _cardinality(self) -> int:
        return self._nodes.numel()

    def _grid_measure(self, n: int) -> Tensor:
        ls = torch.linspace(-1.0, 1.0, n, device=self._device)
        return ls

    def _eval_int_basis(self, ls: Tensor) -> Tensor:
        ls = ls[:, None]
        int_ps = torch.hstack((
            ls,
            -2 ** 0.5 * torch.cos(ls * self._c) / self._c,
            2 ** 0.5 * torch.sin(ls * self._c) / self._c,
            2 ** 0.5 * torch.sin(ls * self._m * torch.pi) / (torch.pi * self._m)
        ))
        return int_ps
    
    def _eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        int_ps = self._eval_int_basis(ls)
        ps = self._basis._eval_basis(ls)
        return int_ps, ps