import abc
import math
from typing import Tuple

import torch
from torch import Tensor

from .domain import Domain


class LinearDomain(Domain, abc.ABC):
    _mean: float
    _dxdl: float

    def _local2approx(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        xs = ls * self._dxdl + self._mean
        dxdls = torch.full_like(ls, self._dxdl)
        return xs, dxdls
    
    def _approx2local(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        ls = (xs - self._mean) / self._dxdl
        dldxs = torch.full_like(xs, 1.0 / self._dxdl)
        return ls, dldxs
    
    def _local2approx_log_density(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        logdxdls = torch.full_like(ls, math.log(self._dxdl))
        logdxdl2s = torch.zeros_like(ls)
        return logdxdls, logdxdl2s
    
    def _approx2local_log_density(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        logdldxs = torch.full_like(xs, -math.log(self._dxdl))
        logdldx2s = torch.zeros_like(xs)
        return logdldxs, logdldx2s