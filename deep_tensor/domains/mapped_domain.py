import abc

import torch 
from torch import Tensor 

from .domain import Domain


class MappedDomain(Domain, abc.ABC):
    
    def __init__(self, scale: float|Tensor = 1.0):
        self.bounds = torch.tensor([-torch.inf, -torch.inf])
        self.scale = torch.tensor(scale)
        return
    
    @property
    def bounds(self) -> Tensor:
        return self._bounds
    
    @bounds.setter
    def bounds(self, value: Tensor) -> None:
        self._bounds = value 
        return