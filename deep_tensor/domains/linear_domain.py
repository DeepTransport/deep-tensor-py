import abc
from typing import Tuple

import torch

from .domain import Domain


class LinearDomain(Domain, abc.ABC):

    @property 
    @abc.abstractmethod
    def mean(self):
        """The midpoint of the approximation domain."""
        return
    
    @property 
    @abc.abstractmethod
    def dxdr(self):
        """The gradient of the mapping from the reference domain to 
        the approximation domain.
        """
        return

    def local2approx(
        self, 
        rs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        xs = rs * self.dxdr + self.mean
        dxdrs = torch.full(rs.shape, self.dxdr)
        return xs, dxdrs
    
    def approx2local(
        self, 
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        rs = (xs - self.mean) / self.dxdr
        drdxs = torch.full(xs.shape, 1.0 / self.dxdr)
        return rs, drdxs
    
    def local2approx_log_density(
        self, 
        rs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        logdxdrs = torch.full(rs.shape, torch.log(self.dxdr))
        logdxdr2s = torch.zeros_like(rs)
        return logdxdrs, logdxdr2s
    
    def approx2local_log_density(
        self, 
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        logdrdxs = torch.full(xs.shape, -torch.log(self.dxdr))
        logdrdx2s = torch.zeros_like(xs)
        return logdrdxs, logdrdx2s