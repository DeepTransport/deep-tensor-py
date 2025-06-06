import torch
from torch import Tensor

from .linear_domain import LinearDomain


class BoundedDomain(LinearDomain):
    r"""Mapping from a bounded domain to $(-1, 1)$.
    
    This class provides a linear mapping from a bounded domain, 
    $(x_{0}, x_{1})$, to $(-1, 1)$.
    
    Parameters
    ----------
    bounds:
        A set of bounds, $(x_{0}, x_{1})$. The default choice is 
        `torch.tensor([-1.0, 1.0])`.
    
    """

    def __init__(self, bounds: Tensor|None = None):
        
        if bounds is None:
            bounds = torch.tensor([-1.0, 1.0])
        
        if not isinstance(bounds, Tensor):
            bounds = torch.tensor(bounds)

        self.check_bounds(bounds)
        self.bounds = bounds.double()
        self.mean = self.bounds.mean()
        self.dxdl = 0.5 * (self.bounds[1] - self.bounds[0])
        return
    
    @property
    def bounds(self) -> Tensor:
        return self._bounds
    
    @bounds.setter
    def bounds(self, value: Tensor) -> None:
        self._bounds = value 
        return

    @property
    def mean(self) -> Tensor:
        return self._mean
    
    @mean.setter
    def mean(self, value: Tensor) -> None:
        self._mean = value 
        return
    
    @property
    def dxdl(self) -> Tensor:
        return self._dxdl
    
    @dxdl.setter
    def dxdl(self, value: Tensor) -> None:
        self._dxdl = value 
        return