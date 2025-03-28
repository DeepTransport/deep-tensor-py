import torch
from torch import Tensor

from .recurr import Recurr
from ..basis_1d import Basis1D


class Legendre(Recurr):
    """Legendre polynomials.
    
    Parameters
    ----------
    order:
        The maximum order of the polynomials.
        
    """

    def __init__(self, order: int):
        n = torch.arange(order+1)
        a = (2*n + 1) / (n + 1)
        b = torch.zeros(n.shape)
        c = n / (n + 1)
        norm = torch.sqrt(2*n + 1)
        Recurr.__init__(self, order, a, b, c, norm)
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

    @property
    def weights(self) -> Tensor:
        return self._weights

    @Basis1D._check_samples
    def eval_measure(self, ls: Tensor) -> Tensor:
        return torch.full(ls.shape, 0.5)
    
    @Basis1D._check_samples
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)

    @Basis1D._check_samples
    def eval_log_measure(self, ls: Tensor) -> Tensor:
        return torch.full(ls.shape, torch.tensor(0.5).log())
        
    @Basis1D._check_samples
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)
    
    def sample_measure(self, n: int) -> Tensor:
        return 2 * torch.rand(n) - 1

    def sample_measure_skip(self, n: int) -> Tensor:
        left  = (torch.min(self.nodes) - 1) / 2
        right = (torch.max(self.nodes) + 1) / 2
        return torch.rand(n) * right-left + left