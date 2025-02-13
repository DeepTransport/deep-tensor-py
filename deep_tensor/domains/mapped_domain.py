import abc

import torch 
from torch import Tensor 

from .domain import Domain


class MappedDomain(Domain, abc.ABC):
    
    def __init__(self, scale: float|Tensor = 1.):  # TODO: check for any other inputs
        self.bounds = torch.tensor([-torch.inf, -torch.inf])
        self.scale = torch.tensor(scale) 
        return
    
    @property
    def left(self) -> Tensor:
        return self._bounds[0]
    
    @property 
    def right(self) -> Tensor:
        return self._bounds[1]
    
    @property
    def bounds(self) -> Tensor:
        return self._bounds
    
    @bounds.setter
    def bounds(self, value: Tensor) -> None:
        self._bounds = value 
        return