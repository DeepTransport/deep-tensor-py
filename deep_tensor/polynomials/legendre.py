import torch

from .recurr import Recurr

DEFAULT_DOMAIN = torch.tensor([-1.0, 1.0])


class Legendre(Recurr):

    def __init__(self, order: int):

        self._domain = DEFAULT_DOMAIN
        self._constant_weight = True

        k = torch.arange(order+1)

        a = (2*k + 1) / (k + 1)
        b = torch.zeros(k.shape)
        c = k / (k+1)
        normalising_const = torch.sqrt(2*k+1)

        super().__init__(order, a, b, c, normalising_const)

    @property
    def domain(self) -> torch.Tensor:
        return self._domain
    
    @property
    def constant_weight(self) -> bool:
        return self._constant_weight

    def eval_measure(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(x.shape, 0.5)
    
    def eval_measure_deriv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def eval_log_measure(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(x.shape, torch.log(torch.tensor(0.5)))
        
    def eval_log_measure_deriv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def sample_measure(self, n: int) -> torch.Tensor:
        return 2 * torch.rand(n) - 1

    def sample_measure_skip(self, n: int) -> torch.Tensor:
        left  = (torch.min(self.nodes) - 1) / 2
        right = (torch.max(self.nodes) + 1) / 2
        return torch.rand(n) * right-left + left