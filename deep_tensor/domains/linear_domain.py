import abc
from typing import Tuple

import torch
from torch import Tensor

from .domain import Domain


class LinearDomain(Domain, abc.ABC):

    @property 
    @abc.abstractmethod
    def mean(self):
        """The midpoint of the approximation domain."""
        return
    
    @property 
    @abc.abstractmethod
    def dxdl(self):
        """The gradient of the mapping from the reference domain to 
        the approximation domain.
        """
        return

    def local2approx(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        xs = ls * self.dxdl + self.mean
        dxdls = torch.full(ls.shape, self.dxdl)
        return xs, dxdls
    
    def approx2local(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        ls = (xs - self.mean) / self.dxdl
        dldxs = torch.full(xs.shape, 1.0 / self.dxdl)
        return ls, dldxs
    
    def local2approx_log_density(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        logdxdls = torch.full(ls.shape, torch.log(self.dxdl))
        logdxdl2s = torch.zeros_like(ls)
        return logdxdls, logdxdl2s
    
    def approx2local_log_density(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        logdldxs = torch.full(xs.shape, -torch.log(self.dxdl))
        logdldx2s = torch.zeros_like(xs)
        return logdldxs, logdldx2s