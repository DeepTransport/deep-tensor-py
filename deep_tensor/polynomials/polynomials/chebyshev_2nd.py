import torch 
from torch import Tensor
from torch.distributions.beta import Beta

from .spectral import Spectral
from ...constants import EPS
from ...tools import check_for_nans


class Chebyshev2nd(Spectral):

    def __init__(self, order: int):

        n = order + 1

        self.domain = torch.tensor([-1.0, 1.0])
        self.order = order 
        self.nodes = torch.cos(torch.pi * torch.arange(1, n+1) / (n+1)).sort()[0]
        self.weights = torch.sin(torch.pi * torch.arange(1, n+1) / (n+1)) * 2 / (n+1)
        
        self.n = torch.arange(self.order+1)
        self.norm = 1.0

        self.__post_init__()
        return
    
    @property 
    def nodes(self) -> Tensor:
        return self._nodes
    
    @nodes.setter 
    def nodes(self, value: Tensor) -> None:
        self._nodes = value 
        return
    
    @property 
    def domain(self) -> Tensor:
        return self._domain 
    
    @domain.setter
    def domain(self, value: Tensor) -> None:
        self._domain = value
        return
    
    @property 
    def weights(self) -> Tensor:
        return self._weights
    
    @weights.setter 
    def weights(self, value: Tensor) -> None:
        self._weights = value 
        return

    @property
    def constant_weight(self) -> bool: 
        return False
    
    def sample_measure(self, n: int) -> Tensor:
        ls = Beta(1.5, 1.5).sample(n)
        return ls
    
    def sample_measure_skip(self, n: int) -> Tensor:
        left = 0.5 * (self.nodes.min() - 1.0)
        right = 0.5 * (self.nodes.max() + 1.0)
        ls = left + torch.rand(n) * (right - left)
        return ls
    
    def eval_measure(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        ws = -ls * ts.sqrt() * 2 / torch.pi 
        return ws
    
    def eval_log_measure(self, ls: Tensor) -> Tensor:
        ts = 1 - ls.square()
        ts[ts < EPS] = EPS
        ws = 0.5 * ts.log() + torch.tensor(2.0/torch.pi).log()
        return ws
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        ts = 1.0 / (1.0 - ls.square())
        check_for_nans(ts)
        ts[ts < EPS] = EPS
        ws = -ls * ts.sqrt() * 2.0 / torch.pi
        return ws
    
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS 
        ws = -ls / ts
        check_for_nans(ws)
        return ws
    
    def eval_basis(self, ls: Tensor) -> Tensor:
        
        thetas = self.l2theta(ls)[:, None]
        ps = torch.sin(thetas * (self.n+1)) / (torch.sin(thetas) / self.norm)

        # Deal with endpoints
        mask_lhs = (ls + 1.0).abs() < EPS
        mask_rhs = (ls - 1.0).abs() < EPS 

        ps[mask_lhs] = self.norm * (self.n+1) * torch.tensor(-1.0).pow(self.n)
        ps[mask_rhs] = self.norm * (self.n+1)
        
        return ps
    
    def eval_basis_deriv(self, ls) -> Tensor:

        thetas = self.l2theta(ls)
        
        ls = ls[:, None]
        thetas = thetas[:, None]

        dpdls = (torch.cos(thetas * (self.n + 1)) * (self.n + 1)
                 - torch.sin(thetas * (self.n + 1)) * (ls / torch.sin(thetas)) / (ls.square() - 1.0))

        dpdls = dpdls * self.norm

        return dpdls
