import abc

import torch 
from torch import Tensor 

from .domain import Domain


class MappedDomain(abc.ABC, Domain):
    
    def __init__(self, scale: float|Tensor = 1.):  # TODO: check for any other inputs
        self.bounds = torch.tensor([-torch.inf, -torch.inf])
        self.scale = torch.tensor(scale) 
        return