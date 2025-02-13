from typing import Tuple

import torch 
from torch import Tensor

from .mapped_domain import MappedDomain
from ..constants import EPS


class AlgebraicDomain(MappedDomain):
    """Maps from an unbounded domain (-inf, -inf) to a bounded domain 
    [-1., 1.].
    """

    def approx2local(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        xs /= self.scale
        ts = 1.0 + xs.square()
        ls = xs * ts.sqrt()
        return ls 
    
    def approx2local_log_density(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        xs /= self.scale 
        ts = 1.0 + xs.square()
        logdldxs = -1.5*torch.log(ts) - torch.log(self.scale)
        logd2ldx2s = (-3. / self.scale) * (xs / ts)
        return logdldxs, logd2ldx2s
    
    def local2approx(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        ls = ls.clamp(-1.+EPS, 1.-EPS)
        ts = 1. - ls.square()
        ts[ts < EPS] = EPS 
        xs = ls * ts.sqrt() * self.scale
        dxdls = ts ** -1.5 * self.scale 
        return xs, dxdls 
    
    def local2approx_log_density(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        ts = 1. - ls.square()
        ts[ts < EPS] = EPS 
        logdxdls = -1.5 * torch.log(ts) + torch.log(self.scale)
        logd2xdl2 = 3. * (ls / ts)
        return logdxdls, logd2xdl2