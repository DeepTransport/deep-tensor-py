import math

import torch
from torch import Tensor

from .spectral import Spectral


class Fourier(Spectral):
    r"""Fourier polynomials.
    
    Parameters
    ----------
    order:
        The number of sine functions the basis is composed of. The 
        total number of basis functions, $n$, is equal to `2*order+2`.
    
    Notes
    -----
    The Fourier basis for the interval $[-1, 1]$, with cardinality $n$, 
    is given by [@Boyd2001; @Cui2022]
    $$
        \left\{1, \sqrt{2}\sin(\pi x), \dots, \sqrt{2}\sin(k \pi x), 
        \sqrt{2}\cos(\pi x), \dots, \sqrt{2}\cos(k \pi x), 
        \sqrt{2}\cos(n \pi x / 2)\right\},
    $$
    where $k = 1, 2, \dots, \tfrac{n}{2}-1$. 
    
    The basis functions are orthonormal with respect to the 
    (normalised) weight function given by
    $$
        \lambda(x) = \frac{1}{2}.
    $$
        
    """

    def __init__(self, order: int):
        
        n_nodes = 2 * order + 2
        n = torch.arange(n_nodes)

        self.order = order
        self._m = order + 1
        self._c = torch.pi * (torch.arange(order) + 1.0)
        self.nodes = 2.0 * (n+1.0) / n_nodes - 1.0
        self.weights = torch.ones_like(self.nodes) / n_nodes

        self.__post_init__()
        self.node2basis[-1] *= 0.5
        return

    @property
    def domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0])
    
    @property
    def constant_weight(self) -> bool:
        return True
    
    @property
    def weights(self) -> Tensor:
        return self._weights
    
    @weights.setter 
    def weights(self, value: Tensor) -> None:
        self._weights = value 
        return

    def sample_measure(self, n: int) -> Tensor:
        return 2.0 * torch.rand(n) - 1.0
    
    def eval_measure(self, ls: Tensor):
        return torch.full(ls.shape, 0.5)
    
    def eval_log_measure(self, ls: Tensor) -> Tensor:
        return torch.full(ls.shape, math.log(0.5))
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)
    
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)
    
    def eval_basis(self, ls: Tensor) -> Tensor:

        self._check_in_domain(ls)
        
        ls = ls[:, None]
        ps = torch.hstack((
            torch.ones_like(ls),
            2 ** 0.5 * torch.sin(ls * self._c),
            2 ** 0.5 * torch.cos(ls * self._c),
            2 ** 0.5 * torch.cos(ls * self._m * torch.pi)
        ))
        return ps
    
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:
        
        self._check_in_domain(ls)

        ls = ls[:, None]
        dpdls = torch.hstack((
            torch.zeros_like(ls),
            2 ** 0.5 * torch.cos(ls * self._c) * self._c,
            -2 ** 0.5 * torch.sin(ls * self._c) * self._c,
            -2 ** 0.5 * torch.sin(ls * self._m * torch.pi) * self._m * torch.pi
        ))
        return dpdls