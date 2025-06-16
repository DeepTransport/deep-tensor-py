import math

import torch 
from torch import Tensor

from .recurr import Recurr
from ...constants import EPS


class Hermite(Recurr):

    def __init__(self, order: int):

        n = torch.arange(order+1)
        a = torch.ones_like(n)
        b = torch.zeros_like(n)
        c = n
        inds = torch.tensor([1, *torch.arange(1, order+1)])
        norm = inds.cumprod(dim=0).reciprocal().sqrt()

        Recurr.__init__(self, order, a, b, c, norm)
        return

    @property
    def domain(self) -> Tensor:
        return torch.tensor([-torch.inf, torch.inf])
    
    @property
    def constant_weight(self) -> bool:
        return False

    @property
    def weights(self) -> Tensor:
        return self._weights
    
    def measure_inverse_cdf(self, zs: Tensor) -> Tensor:
        zs = zs.clamp(EPS, 1.0-EPS)
        ls = (2.0 ** 0.5) * torch.erfinv(2.0*zs-1.0)
        return ls
    
    def sample_measure(self, n: int) -> Tensor:
        return torch.randn(n)
    
    def sample_measure_skip(self, n: int) -> Tensor:
        return self.sample_measure(n)
    
    def eval_measure(self, ls: Tensor) -> Tensor:
        return torch.exp(-0.5 * ls.square()) / torch.tensor(2.0*torch.pi).sqrt()

    def eval_log_measure(self, ls: Tensor) -> Tensor:
        return -0.5 * (ls.square() + math.log(2.0*torch.pi))
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return -ls * torch.exp(-0.5 * ls.square()) / (2.0*torch.pi) ** 0.5
    
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return -ls