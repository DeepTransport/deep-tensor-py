import torch
from torch import Tensor 
from torch.distributions.exponential import Exponential

from .recurr import Recurr
from ...constants import EPS 


class Laguerre(Recurr):

    def __init__(self, order: int):
        
        self.domain = torch.tensor([0., torch.inf])
        self.constant_weight = False 

        n = torch.arange(order+1)
        a = -1. / (n+1)
        b = (2.*n + 1.) / (n + 1.)
        c = n / (n + 1.)
        norm = torch.ones(order+1)
        
        Recurr.__init__(self, order, a, b, c, norm)
    
    def measure_inverse_cdf(self, zs: Tensor) -> Tensor:
        zs = zs.clamp(EPS, 1.-EPS)
        ls = -torch.log(1.-zs)
        return ls 
    
    def sample_measure(self, n: int) -> Tensor:
        return Exponential(rate=1.0).sample(n)
    
    def eval_measure(self, ls: Tensor) -> Tensor:
        return torch.exp(-ls)
    
    def sample_measure_skip(self, n: int) -> Tensor:
        return self.sample_measure(n)

    def eval_log_measure(self, ls: Tensor) -> Tensor:
        return -ls
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return -torch.exp(-ls)
    
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return -torch.ones_like(ls)