import torch 
from torch import Tensor 

from .chebyshev_2nd import Chebyshev2nd
from .trigo_cdf import TrigoCDF


class Chebyshev2ndTrigoCDF(TrigoCDF, Chebyshev2nd):

    def __init__(self, poly: Chebyshev2nd, **kwargs):
        Chebyshev2nd.__init__(self, 2*poly.order)
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

        cdf_ind = torch.arange(1, self.order+3)
        temp = torch.sin(thetas * cdf_ind) / cdf_ind 
        ps = torch.hstack((
            thetas - temp[:, 1][:, None],
            temp[:, :self.order] - temp[:, 2:]
        )) / torch.pi

        return ps
    
    def eval_int_basis_newton(self, thetas: Tensor) -> Tensor:
        
        thetas = thetas[:, None]

        cdf_ind = torch.arange(1, self.order+3)
        temp = torch.sin(cdf_ind * thetas) / cdf_ind 
        
        ps = torch.hstack((
            thetas - temp[:, 1][:, None],
            temp[:, :self.order] - temp[:, 2:]
        )) / torch.pi
        dpdts = (torch.sin(thetas * (self.n+1)) 
                 * torch.sin(thetas) 
                 * (2.0 / torch.pi))

        return ps, dpdts