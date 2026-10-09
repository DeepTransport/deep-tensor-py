from typing import Tuple

import torch 
from torch import Tensor 

from .chebyshev_2nd import Chebyshev2nd
from .trigo_cdf import TrigoCDF


class Chebyshev2ndTrigoCDF(TrigoCDF, Chebyshev2nd):

    def __init__(self, poly: Chebyshev2nd, error_tol: float):
        order = 2 * poly._order
        Chebyshev2nd.__init__(self, order=order, device=poly._device)
        TrigoCDF.__init__(self, error_tol=error_tol, device=poly._device)
        return 
    
    @property
    def _domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0], device=self._device)
    
    @property 
    def _cardinality(self) -> int:
        return self._nodes.numel()
    
    def _eval_int_basis(self, thetas: Tensor) -> Tensor:
        
        thetas = thetas[:, None]

        cdf_ind = torch.arange(1, self._order+3, device=self._device)
        temp = torch.sin(thetas * cdf_ind) / cdf_ind 
        ps = torch.hstack((
            thetas - temp[:, 1][:, None],
            temp[:, :self._order] - temp[:, 2:]
        )) / torch.pi

        return ps
    
    def _eval_int_basis_newton(self, thetas: Tensor) -> Tuple[Tensor, Tensor]:
        
        thetas = thetas[:, None]

        cdf_ind = torch.arange(1, self._order+3, device=self._device)
        temp = torch.sin(cdf_ind * thetas) / cdf_ind 
        
        ps = torch.hstack((
            thetas - temp[:, 1][:, None],
            temp[:, :self._order] - temp[:, 2:]
        )) / torch.pi
        dpdts = (torch.sin(thetas * (self._n+1)) 
                 * torch.sin(thetas) 
                 * (2.0 / torch.pi))

        return ps, dpdts