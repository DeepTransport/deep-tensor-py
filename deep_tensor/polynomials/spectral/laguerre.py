import torch
from torch import Tensor 
from torch.distributions.exponential import Exponential

from .recurr import Recurr
from ..basis_1d import Basis1D
from ...constants import EPS 


class Laguerre(Recurr):

    def __init__(self, order: int):
        n = torch.arange(order+1)
        a = -1.0 / (n+1.0)
        b = (2.0*n + 1.0) / (n+1.0)
        c = n / (n+1.0)
        norm = torch.ones(order+1)
        Recurr.__init__(self, order, a, b, c, norm)
        return
    
    @property
    def domain(self) -> Tensor:
        return torch.tensor([0.0, torch.inf])
    
    @property
    def constant_weight(self) -> bool:
        return False
    
    @property 
    def nodes(self) -> Tensor:
        return self._nodes

    @property
    def weights(self) -> Tensor:
        return self._weights

    def measure_inverse_cdf(self, zs: Tensor) -> Tensor:
        zs = zs.clamp(EPS, 1.0-EPS)
        ls = -torch.log(1.0-zs)
        return ls 
    
    def sample_measure_skip(self, n: int) -> Tensor:
        return self.sample_measure(n)
    
    def sample_measure(self, n: int) -> Tensor:
        return Exponential(rate=1.0).sample(n)
    
    @Basis1D._check_samples
    def eval_measure(self, ls: Tensor) -> Tensor:
        return torch.exp(-ls)

    @Basis1D._check_samples
    def eval_log_measure(self, ls: Tensor) -> Tensor:
        return -ls
    
    @Basis1D._check_samples
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return -torch.exp(-ls)
    
    @Basis1D._check_samples
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return -torch.ones_like(ls)