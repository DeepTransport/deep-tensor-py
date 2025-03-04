import torch
from torch import Tensor

from .spectral import Spectral


class Fourier(Spectral):
    """Fourier polynomials.
    
    Parameters
    ----------
    order:
        The maximum order of the polynomials.
        
    """

    def __init__(self, order: int):
        
        n_nodes = 2 * order + 2
        n = torch.arange(n_nodes)

        self.order = order
        self.m = self.order + 1
        self._nodes = torch.sort((2.0/n_nodes) * (n+1) - 1).values
        self.weights = torch.ones_like(self.nodes) / n_nodes
        self.c = (torch.arange(self.order)+1) * torch.pi

        self.__post_init__()
        # TODO: figure out what's going on here
        self.node2basis[-1] *= 0.5
        return

    @property
    def domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0])
    
    @property
    def constant_weight(self) -> bool:
        return True
    
    @property 
    def nodes(self) -> Tensor:
        return self._nodes
    
    @nodes.setter 
    def nodes(self, value: Tensor) -> None:
        self._nodes = value 
        return

    @property
    def weights(self) -> Tensor:
        return self._weights
    
    @weights.setter 
    def weights(self, value: Tensor) -> None:
        self._weights = value 
        return

    def sample_measure(self, n: int) -> Tensor:
        return torch.rand(n) * 2 - 1
    
    def sample_measure_skip(self, n: int) -> Tensor:
        return self.sample_measure(n)
    
    def eval_measure(self, ls: Tensor):
        return torch.full(ls.shape, 0.5)
    
    def eval_log_measure(self, ls: Tensor) -> Tensor:
        return torch.full(ls.shape, torch.tensor(0.5).log())
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)
    
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)
    
    def eval_basis(self, ls: Tensor) -> Tensor:

        ls = ls[:, None]
        ps = torch.hstack((
            torch.ones_like(ls),
            2 ** 0.5 * torch.sin(ls * self.c),
            2 ** 0.5 * torch.cos(ls * self.c),
            2 ** 0.5 * torch.cos(ls * self.m * torch.pi)
        ))
        return ps
    
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:

        ls = ls[:, None]
        dpdls = torch.hstack((
            torch.zeros_like(ls),
            2 ** 0.5 * torch.cos(ls * self.c) * self.c,
            -2 ** 0.5 * torch.sin(ls * self.c) * self.c,
            -2 ** 0.5 * torch.sin(ls * self.m * torch.pi) * self.m * torch.pi
        ))
        return dpdls