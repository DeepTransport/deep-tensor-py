import torch
from torch import Tensor

from .recurr import Recurr


class Legendre(Recurr):

    def __init__(self, order: int):

        self.domain = torch.tensor([-1.0, 1.0])
        self.constant_weight = True

        n = torch.arange(order+1)
        a = (2*n + 1) / (n + 1)
        b = torch.zeros(n.shape)
        c = n / (n + 1)
        norm = torch.sqrt(2*n + 1)

        Recurr.__init__(self, order, a, b, c, norm)
        return

    @property
    def domain(self) -> Tensor:
        return self._domain
    
    @domain.setter
    def domain(self, value: Tensor) -> None:
        self._domain = value
        return
    
    @property
    def constant_weight(self) -> bool:
        return self._constant_weight
    
    @constant_weight.setter
    def constant_weight(self, value: Tensor) -> None:
        self._constant_weight = value
        return
    
    @property 
    def nodes(self) -> Tensor:
        return self._nodes

    @property
    def weights(self) -> Tensor:
        return self._weights

    def eval_measure(self, ls: Tensor) -> Tensor:
        return torch.full(ls.shape, 0.5)
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)

    def eval_log_measure(self, ls: Tensor) -> Tensor:
        return torch.full(ls.shape, torch.tensor(0.5).log())
        
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)
    
    def sample_measure(self, n: int) -> Tensor:
        return 2 * torch.rand(n) - 1

    def sample_measure_skip(self, n: int) -> Tensor:
        left  = (torch.min(self.nodes) - 1) / 2
        right = (torch.max(self.nodes) + 1) / 2
        return torch.rand(n) * right-left + left