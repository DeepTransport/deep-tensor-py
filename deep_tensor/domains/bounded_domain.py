from typing import List

from torch import Tensor

from .linear_domain import LinearDomain


class BoundedDomain(LinearDomain):
    r"""Mapping from a bounded domain to $[-1, 1]$.
    
    This class provides a linear mapping from a bounded domain, 
    $[x_{0}, x_{1}]$, to $[-1, 1]$.
    
    Parameters
    ----------
    bounds:
        A set of bounds, $[x_{0}, x_{1}]$. The default choice is 
        `[-1.0, 1.0]` (in which case the mapping is the identity 
        mapping).
    
    """

    def __init__(self, bounds: List | None = None):  
        if bounds is None:
            bounds = [-1.0, 1.0]
        if isinstance(bounds, Tensor):
            bounds = bounds.tolist()
        self.check_bounds(bounds)
        self.bounds = bounds
        self.mean = 0.5 * (bounds[0] + bounds[1])
        self.dxdl = 0.5 * (bounds[1] - bounds[0])
        return
    
    @property
    def bounds(self) -> List:
        return self._bounds
    
    @bounds.setter
    def bounds(self, value: List) -> None:
        self._bounds = value 
        return

    @property
    def mean(self) -> float:
        return self._mean
    
    @mean.setter
    def mean(self, value: float) -> None:
        self._mean = value 
        return
    
    @property
    def dxdl(self) -> float:
        return self._dxdl
    
    @dxdl.setter
    def dxdl(self, value: float) -> None:
        self._dxdl = value 
        return