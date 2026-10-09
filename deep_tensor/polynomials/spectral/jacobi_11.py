import math

import torch 
from torch import Tensor 
from torch.distributions.beta import Beta

from .recurr import Recurr


class Jacobi11(Recurr):

    def __init__(
        self, 
        order: int, 
        device: torch.device = torch.get_default_device()
    ):
        k = torch.arange(order+1, device=device)
        a = (2*k+3) * (k+2) / (k+1) / (k+3)
        b = torch.zeros_like(k)
        c = (k+2)/(k+3)
        norm = ((2.0*k+3.0) * (k+2.0) / (8.0 * (k+1.0)) * (4/3)).sqrt()
        self._device = device
        Recurr.__init__(self, order, a, b, c, norm, self._device)
        return
    
    @property 
    def _domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0], device=self._device)
    
    @property
    def _constant_weight(self) -> bool:
        return False
    
    def _sample_measure(self, n: int) -> Tensor:
        ls = Beta(2.0, 2.0).sample((n,)).to(self._device)
        ls = (2.0 * ls) - 1.0
        return ls
    
    def _sample_measure_skip(self, n: int) -> Tensor:
        l0 = 0.5 * (self._nodes.min() - 1.0)
        l1 = 0.5 * (self._nodes.max() + 1.0)
        ls = torch.rand(n, device=self._device) * (l1-l0) + l0
        return ls
    
    def _eval_measure(self, ls: Tensor) -> Tensor:
        ws = 0.75 * (1.0 - ls.square())
        return ws
    
    def _eval_log_measure(self, ls: Tensor) -> Tensor:
        ws = (1.0 - ls.square()).log() + math.log(0.75)
        return ws
    
    def _eval_measure_deriv(self, ls: Tensor) -> Tensor:
        ws = -1.5 * ls
        return ws
    
    def _eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        raise NotImplementedError()