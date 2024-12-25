import abc

import torch

from .basis_1d import Basis1D


class Piecewise(Basis1D, abc.ABC):

    def __init__(self, order: int, num_elems: int):

        self._domain = torch.tensor([-1.0, 1.0])
        self._grid = torch.linspace(*self.domain, num_elems+1)
        self._constant_weight = True

        self.order = order 
        self.num_elems = num_elems
        
        self.elem_size = self.grid[1] - self.grid[0]
        self.domain_size = self.domain[1] - self.domain[0]

        return
    
    @property
    def domain(self) -> torch.Tensor:
        return self._domain 
    
    @property 
    def grid(self) -> torch.Tensor:
        return self._grid
    
    @property 
    def constant_weight(self) -> bool:
        return self._constant_weight

    def sample_measure(self, n: int) -> torch.Tensor:
        return self.domain[0] + self.domain_size * torch.rand(n)

    def sample_measure_skip(self, n: int) -> torch.Tensor:
        return self.sample_measure(n)

    def eval_measure(self, ls: torch.Tensor) -> torch.Tensor:
        return torch.full(ls.shape, 1.0 / self.domain_size)

    def eval_log_measure(self, ls: torch.Tensor) -> torch.Tensor:
        return torch.full(ls.shape, -torch.log(self.domain_size))

    def eval_measure_deriv(obj, ls: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(ls)

    def eval_log_measure_deriv(self, ls: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(ls)