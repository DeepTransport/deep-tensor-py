import math

import torch 
from torch import Tensor
from torch.distributions.beta import Beta

from .spectral import Spectral
from ...constants import EPS
from ...tools import check_finite


class Chebyshev2nd(Spectral):
    r"""Chebyshev polynomials of the second kind.
    
    Parameters
    ----------
    order:
        The maximum order of the polynomials.

    Notes
    -----
    The (normalised) Chebyshev polynomials of the second kind, defined 
    on $(-1, 1)$, are given by [@Boyd2001]
    $$
        p_{k}(x) = \frac{\sin((k+1)\arccos(x))}{\sin{(\arccos(x))}}, 
            \qquad k = 0, 1, \dots, n.
    $$
    The polynomials are orthonormal with respect to the (normalised) 
    weighting function given by
    $$
        \lambda(x) = \frac{2\sqrt{1-x^{2}}}{\pi}.
    $$
        
    """

    def __init__(
        self, 
        order: int, 
        device: torch.device = torch.get_default_device()
    ):

        n = order + 1

        self._order = order 
        self._device = device
        self._nodes = torch.cos(torch.pi * torch.arange(1, n+1, device=self._device) / (n+1)).sort().values
        self._weights = torch.sin(torch.pi * torch.arange(1, n+1, device=self._device) / (n+1)).square() * 2 / (n+1)
        
        self._n = torch.arange(self._order+1, device=self._device)
        self._norm = 1.0

        self.__post_init__(self._device)
        return
    
    @property 
    def _domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0], device=self._device)
    
    @property
    def _constant_weight(self) -> bool: 
        return False
    
    def _sample_measure(self, n: int) -> Tensor:
        ls = Beta(1.5, 1.5).sample((n,)).to(self._device)
        ls = 2.0 * ls - 1.0
        return ls
    
    def _eval_measure(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ts = 1.0 - ls.square()
        ws = 2.0 * ts.sqrt() / torch.pi 
        return ws
    
    def _eval_log_measure(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ls = ls.clamp(self._domain[0]+EPS, self._domain[1]-EPS)
        ts = 1.0 - ls.square()
        logws = 0.5 * ts.log() + math.log(2.0/torch.pi)
        return logws
    
    def _eval_measure_deriv(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ls = ls.clamp(self._domain[0]+EPS, self._domain[1]-EPS)
        ts = 1.0 / (1.0 - ls.square())
        check_finite(ts)
        dwdls = -2.0 * ls * ts.sqrt() / torch.pi
        return dwdls
    
    def _eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        ws = -ls / ts
        check_finite(ws)
        return ws
    
    def _eval_basis(self, ls: Tensor) -> Tensor:

        self._check_in_domain(ls)
        
        thetas = self._l2theta(ls)[:, None]
        sin_thetas = thetas.sin()
        sin_thetas[sin_thetas.abs() < EPS] = EPS

        ps = self._norm * torch.sin(thetas * (self._n+1)) / sin_thetas

        # Deal with endpoints
        mask_lhs = (ls + 1.0).abs() < EPS
        mask_rhs = (ls - 1.0).abs() < EPS
        ps[mask_lhs] = self._norm * (self._n+1) * torch.tensor(-1.0, device=self._device).pow(self._n)
        ps[mask_rhs] = self._norm * (self._n+1)
        check_finite(ps)
        return ps
    
    def _eval_basis_deriv(self, ls: Tensor) -> Tensor:

        self._check_in_domain(ls)

        thetas = self._l2theta(ls)[:, None]
        sin_thetas = thetas.sin()
        sin_thetas[sin_thetas.abs() < EPS] = EPS
        ls = ls[:, None]
        
        ts = ls.square() - 1.0
        ts[ts > -EPS] = -EPS

        dpdls = self._norm * ((torch.cos(thetas * (self._n+1)) * (self._n+1)
                              - torch.sin(thetas * (self._n+1)) * (ls / sin_thetas)) / ts)
        check_finite(dpdls)
        return dpdls