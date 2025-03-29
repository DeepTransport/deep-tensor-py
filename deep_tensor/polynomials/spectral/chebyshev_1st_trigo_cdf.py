import torch 
from torch import Tensor 

from .chebyshev_1st import Chebyshev1st
from .trigo_cdf import TrigoCDF


class Chebyshev1stTrigoCDF(TrigoCDF, Chebyshev1st):

    def __init__(self, poly: Chebyshev1st, **kwargs):
        Chebyshev1st.__init__(self, 2*poly.order)
        TrigoCDF.__init__(self, **kwargs)
        return
    
    @property
    def domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0])
    
    @property
    def node2basis(self) -> Tensor:
        return self._node2basis
    
    @node2basis.setter
    def node2basis(self, value: Tensor) -> None:
        self._node2basis = value 
        return
    
    @property
    def basis2node(self) -> Tensor:
        return self._basis2node
    
    @basis2node.setter
    def basis2node(self, value: Tensor) -> None:
        self._basis2node = value 
        return
    
    @property
    def nodes(self) -> Tensor:
        return self._nodes
    
    @nodes.setter 
    def nodes(self, value: Tensor) -> None:
        self._nodes = value 
        return

    @property 
    def cardinality(self) -> int:
        return self.nodes.numel()
    
    def eval_int_basis(self, thetas: Tensor) -> Tensor:
        
        thetas = thetas[:, None]
        
        # Cui et al, 2023
        int_pws = torch.hstack((
            thetas / torch.pi, 
            torch.tensor(2.0).sqrt() / (torch.pi * self.n[1:]) 
                * torch.sin(thetas * self.n[1:])
        ))
        return int_pws
    
    def eval_int_basis_newton(self, thetas: Tensor) -> Tensor:
        int_pws = self.eval_int_basis(thetas)
        thetas = thetas[:, None]
        derivs = self.norm * torch.cos(thetas * self.n) / torch.pi
        return int_pws, derivs