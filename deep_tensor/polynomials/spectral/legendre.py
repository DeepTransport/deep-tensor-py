import math

import torch
from torch import Tensor

from .recurr import Recurr


class Legendre(Recurr):
    r"""Legendre polynomials.
    
    Parameters
    ----------
    order:
        The maximum order of the polynomials, $n$.

    Notes
    -----
    The Legendre polynomials, defined on $(-1, 1)$, are given by the 
    recurrence relation [@Boyd2001]
    $$
        (k+1)\hat{p}_{k+1}(x) = (2k+1)x\hat{p}_{k}(x) - k\hat{p}_{k-1}(x), 
            \qquad k = 1, 2, \dots, n-1,
    $$ 
    where $\hat{p}_{0}(x) = 1, \hat{p}_{1}(x) = x$. The corresponding 
    normalised polynomials are given by
    $$
        p_{k}(x) := \frac{\hat{p}_{k}(x)}{2k+1}, 
            \qquad k = 0, 1, \dots, n.
    $$

    The polynomials are orthonormal with respect to the (normalised) 
    weighting function given by
    $$
        \lambda(x) = \frac{1}{2}.
    $$

    We use Chebyshev polynomials of the second kind to represent the 
    (conditional) CDFs corresponding to the Legendre representation of 
    (the square root of) the target density function.
        
    """

    def __init__(
        self, 
        order: int, 
        device: torch.device = torch.device("cpu")
    ):
        self.device = device
        n = torch.arange(order+1, device=self.device)
        a = (2*n + 1) / (n + 1)
        b = torch.zeros_like(n)
        c = n / (n + 1)
        norm = torch.sqrt(2*n + 1)
        Recurr.__init__(self, order, a, b, c, norm, self.device)
        return

    @property
    def domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0], device=self.device)
    
    @property
    def constant_weight(self) -> bool:
        return True
    
    @property 
    def nodes(self) -> Tensor:
        return self._nodes

    @property
    def weights(self) -> Tensor:
        return self._weights
    
    def sample_measure(self, n: int) -> Tensor:
        return 2.0 * torch.rand(n, device=self.device) - 1.0

    def eval_measure(self, ls: Tensor) -> Tensor:
        return torch.full_like(ls, 0.5)
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)

    def eval_log_measure(self, ls: Tensor) -> Tensor:
        return torch.full_like(ls, math.log(0.5))
        
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)