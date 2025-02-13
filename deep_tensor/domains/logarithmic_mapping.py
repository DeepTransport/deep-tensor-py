from typing import Tuple 

import torch 
from torch import Tensor

from .mapped_domain import MappedDomain
from ..constants import EPS 


class LogarithmicMapping(MappedDomain):

    def approx2local(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        ls = torch.atan(xs / self.scale)
        ls = ls.clamp(-1.+EPS, 1.-EPS)
        ts = 1. - ls.square()
        ts[ts < EPS] = EPS 
        dldxs = ts / self.scale
        return ls, dldxs
    
    def approx2local_log_density(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        ls = torch.atan(xs / self.scale) 
        ls = ls.clamp(-1. + EPS, 1. - EPS)
        logdldxs = torch.log(1. - ls.square()) - self.scale.log()
        logd2ldx2s = (-2. / self.scale) * ls 
        return logdldxs, logd2ldx2s
    
    def local2approx(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        xs = torch.atan(ls) * self.scale 
        ts = 1. - ls.square()
        ts[ts < EPS] = EPS 
        dxdls = self.scale / ts
        return xs, dxdls
    
    def local2approx_log_density(self, ls: Tensor) -> Tensor:
        ts = 1. - ls.square()
        ts[ts < EPS] = EPS 
        logdxdls = -torch.log(ts) + self.scale.log()
        logd2xdl2s = 2. * (ls / ts)
        return logdxdls, logd2xdl2s