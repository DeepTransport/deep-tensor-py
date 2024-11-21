import abc
from typing import Tuple

import torch

from .domain import Domain


class LinearDomain(Domain, abc.ABC):

    @property 
    @abc.abstractmethod
    def mean(self):
        return
    
    @property 
    @abc.abstractmethod
    def dxdz(self):
        return

    def reference2domain(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x = z * self.dxdz + self.mean
        dxdz = torch.full(z.shape, self.dxdz)
        return x, dxdz
    
    def domain2reference(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        z = (x-self.mean) / self.dxdz
        dzdx = torch.full(x.shape, 1.0 / self.dxdz)
        return z, dzdx
    
    def reference2domain_log_density(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        logdxdz = torch.full(z.shape, torch.log(self.dxdz))
        logdxdz2 = torch.zeros_like(z)
        return logdxdz, logdxdz2
    
    def domain2reference_log_density(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        logdzdx = torch.full(x.shape, -torch.log(self.dxdz))
        logdzdx2 = torch.zeros_like(x)
        return logdzdx, logdzdx2