import torch

from .spectral import Spectral


class Fourier(Spectral):

    def __init__(self, order: int):

        self._domain = torch.tensor([-1.0, 1.0])
        self._constant_weight = True
        
        self.order = order
        self.m = self.order + 1

        num_nodes = 2 + self.order * 2
        n = torch.arange(num_nodes)

        self._nodes = torch.sort((2.0/num_nodes) * (n+1) - 1).values
        self._weights = torch.ones_like(self.nodes) / num_nodes

        self.c = (torch.arange(self.order)+1) * torch.pi

        self.post_construction()
        self.node2basis[-1] *= 0.5  # TODO: figure out what's going on here / whether this has any effect on the TT building

        return

    @property
    def domain(self) -> torch.Tensor:
        return self._domain
    
    @property
    def constant_weight(self) -> bool:
        return self._constant_weight
    
    @property 
    def nodes(self) -> torch.Tensor:
        return self._nodes

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    def sample_measure(self, n: int) -> torch.Tensor:
        return torch.rand(n) * 2 - 1
    
    def sample_measure_skip(self, n: int) -> torch.Tensor:
        return self.sample_measure(n)
    
    def eval_measure(self, x: torch.Tensor):
        return 0.5 * torch.ones_like(x)
    
    def eval_log_measure(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(x.shape, torch.tensor(0.5).log())
    
    def eval_measure_deriv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def eval_log_measure_deriv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def eval_basis(self, us: torch.Tensor) -> torch.Tensor:

        tmp = torch.outer(us, self.c)
        basis_vals = torch.hstack((
            torch.ones((us.numel(), 1)),
            2 ** 0.5 * torch.sin(tmp),
            2 ** 0.5 * torch.cos(tmp),
            2 ** 0.5 * torch.cos(us[:, None] * self.m * torch.pi)
        ))
        
        return basis_vals
    
    def eval_basis_deriv(self, x):
        raise NotImplementedError()
     
    def eval(self, coeffs, xs):
        raise NotImplementedError()