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

    def __init__(
        self, 
        order: int, 
        device: torch.device = torch.get_default_device()
    ):

        self._order = order 
        self._device = device

        num_nodes = 2 * order + 2
        n = torch.arange(num_nodes, device=self._device)
        self._m = order + 1
        self._c = torch.pi * (torch.arange(order, device=self._device) + 1.0)
        self._nodes = 2.0 * (n + 1.0) / num_nodes - 1.0
        self._weights = torch.ones_like(self._nodes) / num_nodes

        self.__post_init__(self._device)
        self._node2basis[-1] *= 0.5
        return

    @property
    def _domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0], device=self._device)
    
    @property
    def _constant_weight(self) -> bool:
        return True
    
    def _sample_measure(self, n: int) -> Tensor:
        return 2.0 * torch.rand(n, device=self._device) - 1.0
    
    def _eval_measure(self, ls: Tensor):
        return torch.full_like(ls, 0.5)
    
    def _eval_log_measure(self, ls: Tensor) -> Tensor:
        return torch.full_like(ls, math.log(0.5))
    
    def _eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)
    
    def _eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)
    
    def _eval_basis(self, ls: Tensor) -> Tensor:

        self._check_in_domain(ls)
        
        ls = ls[:, None]
        ps = torch.hstack((
            torch.ones_like(ls),
            2 ** 0.5 * torch.sin(ls * self._c),
            2 ** 0.5 * torch.cos(ls * self._c),
            2 ** 0.5 * torch.cos(ls * self._m * torch.pi)
        ))
        return ps
    
    def _eval_basis_deriv(self, ls: Tensor) -> Tensor:
        
        self._check_in_domain(ls)

        ls = ls[:, None]
        dpdls = torch.hstack((
            torch.zeros_like(ls),
            2 ** 0.5 * torch.cos(ls * self._c) * self._c,
            -2 ** 0.5 * torch.sin(ls * self._c) * self._c,
            -2 ** 0.5 * torch.sin(ls * self._m * torch.pi) * self._m * torch.pi
        ))
        return dpdls