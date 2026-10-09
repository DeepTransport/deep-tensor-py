import math

import torch
from torch import Tensor

from .spectral import Spectral 
from ...constants import EPS
from ...tools import check_finite


class Chebyshev1st(Spectral):
    r"""Chebyshev polynomials of the first kind.

    Parameters
    ----------
    order:
        The maximum order of the polynomials.

    Notes
    -----
    The (normalised) Chebyshev polynomials of the first kind, defined 
    on $(-1, 1)$, are given by [@Boyd2001; @Cui2023b]
    $$
    \begin{align}
        p_{0}(x) &= 1, \\
        p_{k}(x) &= \sqrt{2}\cos(k\arccos(x)), 
            \qquad k = 1, 2, \dots, n.
    \end{align}
    $$
    The polynomials are orthonormal with respect to the (normalised) 
    weighting function given by
    $$
        \lambda(x) = \frac{1}{\pi\sqrt{1-x^{2}}}.
    $$

    """

    def __init__(
        self, 
        order: int, 
        device: torch.device = torch.get_default_device()
    ):

        self._order = order
        self._device = device
        self._n = torch.arange(self._order+1, device=self._device)
        nodes = torch.cos(torch.pi * (self._n+0.5) / (self._order+1))
        self._nodes = nodes.sort().values
        self._weights = torch.ones_like(self._nodes) / (self._order+1)

        self._norm = torch.hstack((
            torch.tensor([1.0], device=self._device), 
            torch.full((self._order,), math.sqrt(2.0), device=self._device)
        ))

        self.__post_init__(self._device)
        return
    
    @property 
    def _domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0], device=self._device)
    
    @property
    def _constant_weight(self) -> bool: 
        return False

    def _eval_measure(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return 1.0 / (torch.pi * ts**0.5)
    
    def _eval_measure_deriv(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return (ls / torch.pi) * ts ** -1.5

    def _eval_log_measure(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return -0.5 * torch.log(ts) - math.log(torch.pi)

    def _eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return ls / ts

    def _sample_measure(self, n: int) -> Tensor:
        zs = torch.rand(n, device=self._device)
        samples = torch.sin(torch.pi * (zs - 0.5))
        return samples
    
    def _eval_basis(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        thetas = self._l2theta(ls)[:, None]
        ps = self._norm * torch.cos(thetas * self._n)
        return ps
    
    def _eval_basis_deriv(self, ls: Tensor) -> Tensor:

        self._check_in_domain(ls)

        thetas = self._l2theta(ls)[:, None]
        sin_thetas = thetas.sin()
        sin_thetas[sin_thetas.abs() < EPS] = EPS

        dpdls = self._norm * torch.hstack((
            torch.zeros_like(thetas),
            self._n[1:] * torch.sin(thetas * self._n[1:]) / sin_thetas
        ))
        check_finite(dpdls)
        return dpdls 