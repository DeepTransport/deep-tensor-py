from typing import Tuple

import torch
from torch import Tensor

from .spectral_cdf import SpectralCDF
from .fourier import Fourier


class FourierCDF(SpectralCDF, Fourier):

    def __init__(self, poly: Fourier, **kwargs):
        Fourier.__init__(self, 2*poly.order)
        SpectralCDF.__init__(self, **kwargs)
        return
    
    @property
    def domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0])
    
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
    
    @nodes.setter
    def nodes(self, value: Tensor) -> None:
        self._nodes = value 
        return None

    @property 
    def cardinality(self) -> int:
        return self.nodes.numel()

    def grid_measure(self, n: int) -> Tensor:
        ls = torch.linspace(*self.domain, n)
        return ls

    def eval_int_basis(self, ls: Tensor) -> Tensor:
        ls = ls[:, None]
        int_ps = torch.hstack((
            ls,
            -2 ** 0.5 * torch.cos(ls * self.c) / self.c,
            2 ** 0.5 * torch.sin(ls * self.c) / self.c,
            2 ** 0.5 * torch.sin(ls * self.m * torch.pi) / (torch.pi * self.m)
        ))
        return int_ps
    
    def eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        int_ps = self.eval_int_basis(ls)
        ps = self.eval_basis(ls)
        return int_ps, ps