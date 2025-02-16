import torch 
from torch import Tensor
from torch.distributions.beta import Beta

from .spectral import Spectral
from ...constants import EPS


class Chebyshev2ndUnweighted(Spectral):

    def __init__(self, order: int):

        self.order = order
        self.n = torch.arange(self.order+1)

        self._domain = torch.tensor([-1.0, 1.0])

        self._nodes = torch.cos(torch.pi * (self.n+1) / (self.order+2))
        self._nodes = torch.sort(self._nodes)[0]
        self._weights = torch.sin(torch.pi * (self.n+1) / (self.order+2)) ** 2 * 2 / (self.order+2)
        self._constant_weight = False

        self.normalising = torch.tensor([1.0])

        self.__post_init__()
        return
    
    @property
    def domain(self) -> Tensor:
        return self._domain

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
    
    @property
    def constant_weight(self) -> bool:
        return self._constant_weight

    def sample_measure(self, n: int) -> Tensor:
        
        beta = Beta(1.5, 1.5)
        rs = beta.sample((n, self.order))
        rs = 2.0 * rs - 1.0
        return rs
    
    def sample_measure_skip(self, n: int) -> Tensor:
        r0 = 0.5 * (torch.min(self.nodes) - 1.0)
        r1 = 0.5 * (torch.max(self.nodes) + 1.0)
        rs = torch.rand(n) * (r1-r0) + r0
        return rs
    
    def eval_measure(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = 0.0
        ws = 2.0 * torch.sqrt(ts) / torch.pi 
        return ws
    
    def eval_log_measure(self, rs: Tensor) -> Tensor:
        ts = 1.0 - rs.square()
        ts[ts < EPS] = 0.0
        ws = 0.5*torch.log(ts) + (torch.tensor(2.0) / torch.pi).log()
        return ws
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = 0.0
        ws = -ls * ts.sqrt() * 2.0 / torch.pi
        return ws
    
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        raise NotImplementedError()

    def eval_basis(self, ls: Tensor) -> Tensor:

        thetas = self.l2theta(ls)
        fs = (torch.sin(torch.outer(self.n+1, thetas)) 
              * (self.normalising / torch.sin(thetas))).T

        if (mask_lhs := torch.abs(ls+1) < EPS).sum() > 0:
            fs[mask_lhs, :] = (self.n+1) * self.normalising * torch.pow(-1.0, self.n)
        
        if (mask_rhs := torch.abs(ls-1) < EPS).sum() > 0:
            fs[mask_rhs, :] = (self.n+1) * self.normalising
        
        return fs

    def eval_basis_deriv(self, ls: Tensor) -> Tensor:
        raise NotImplementedError()