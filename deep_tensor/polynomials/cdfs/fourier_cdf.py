from typing import Tuple

import torch
from torch import Tensor

from .spectral_cdf import SpectralCDF
from ..polynomials.fourier import Fourier
from ...constants import EPS


class FourierCDF(SpectralCDF, Fourier):

    def __init__(self, poly: Fourier, **kwargs):
        Fourier.__init__(self, 2 * poly.order)
        SpectralCDF.__init__(self, **kwargs)
        return
    
    @property
    def domain(self) -> Tensor:
        return self._domain 
    
    @property
    def node2basis(self) -> Tensor:
        return self._node2basis
    
    @node2basis.setter
    def node2basis(self, value: Tensor) -> None:
        self._node2basis = value 
        return None
    
    @property
    def basis2node(self) -> Tensor:
        return self._basis2node
    
    @basis2node.setter
    def basis2node(self, value: Tensor) -> None:
        self._basis2node = value 
        return None
    
    @property
    def nodes(self) -> Tensor:
        return self._nodes

    @property 
    def cardinality(self) -> int:
        return self.nodes.numel()

    def grid_measure(self, n: int) -> Tensor:
        us = torch.linspace(*self.domain, n)
        us[0] = self.domain[0] - EPS 
        us[-1] = self.domain[-1] + EPS
        return us

    def eval_int_basis(self, us: Tensor) -> Tensor:

        tmp = torch.outer(us, self.c)
        
        int_basis_vals = torch.hstack((
            us[:, None],
            -(2.0 ** 0.5 / self.c) * torch.cos(tmp),
            (2.0 ** 0.5 / self.c) * torch.sin(tmp),
            ((2.0 ** 0.5 / (torch.pi * self.m)) 
             * torch.sin(us[:, None] * self.m * torch.pi))
        ))
        
        return int_basis_vals
    
    def eval_int_basis_newton(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        basis_vals = self.eval_int_basis(us)
        deriv_vals = self.eval_basis(us)
        return basis_vals, deriv_vals