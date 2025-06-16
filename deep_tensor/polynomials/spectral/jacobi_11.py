import torch 
from torch import Tensor 
from torch.distributions.beta import Beta

from .recurr import Recurr


class Jacobi11(Recurr):

    def __init__(self, order: int):
        k = torch.arange(order+1)
        a = (2*k+3) * (k+2) / (k+1) / (k+3)
        b = torch.zeros_like(k)
        c = (k+2)/(k+3)
        norm = ((2.0*k+3.0) * (k+2.0) / (8.0 * (k+1.0)) * (4/3)).sqrt()
        Recurr.__init__(self, order, a, b, c, norm)
        return
    
    @property
    def weights(self) -> Tensor:
        return self._weights

    @property 
    def domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0])
    
    @property
    def constant_weight(self) -> bool:
        return False
    
    def sample_measure(self, n: int) -> Tensor:
        beta = Beta(2.0, 2.0)
        ls = beta.sample((n,))
        ls = (2.0 * ls) - 1.0
        return ls
    
    def sample_measure_skip(self, n: int) -> Tensor:
        l0 = 0.5 * (self.nodes.min() - 1.0)
        l1 = 0.5 * (self.nodes.max() + 1.0)
        ls = torch.rand(n) * (l1-l0) + l0
        return ls
    
    def eval_measure(self, ls: Tensor) -> Tensor:
        ws = 0.75 * (1.0 - ls.square())
        return ws
    
    def eval_log_measure(self, ls: Tensor) -> Tensor:
        ws = (1.0 - ls.square()).log() + torch.tensor(0.75).log()
        return ws
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        ws = -1.5 * ls
        return ws
    
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        raise NotImplementedError()