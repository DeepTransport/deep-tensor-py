from typing import Tuple 

import torch 
from torch import Tensor

from .mapped_domain import MappedDomain
from ..constants import EPS 
from ..tools import check_for_nans


class LogarithmicMapping(MappedDomain):

    def approx2local(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        ls = torch.tanh(xs / self.scale)
        ls = ls.clamp(-1.+EPS, 1.-EPS)
        ts = 1. - ls.square()
        ts[ts < EPS] = EPS 
        dldxs = ts / self.scale
        check_for_nans(ls)
        check_for_nans(dldxs)
        return ls, dldxs
    
    def approx2local_log_density(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        ls = torch.tanh(xs / self.scale) 
        ls = ls.clamp(-1. + EPS, 1. - EPS)
        logdldxs = torch.log(1. - ls.square()) - self.scale.log()
        logd2ldx2s = (-2. / self.scale) * ls 
        check_for_nans(logdldxs)
        check_for_nans(logd2ldx2s)
        return logdldxs, logd2ldx2s
    
    def local2approx(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        xs = torch.atanh(ls) * self.scale 
        ts = 1. - ls.square()
        ts[ts < EPS] = EPS 
        dxdls = self.scale / ts
        check_for_nans(xs)
        check_for_nans(dxdls)
        return xs, dxdls
    
    def local2approx_log_density(self, ls: Tensor) -> Tensor:
        ts = 1. - ls.square()
        ts[ts < EPS] = EPS 
        logdxdls = -torch.log(ts) + self.scale.log()
        logd2xdl2s = 2. * (ls / ts)
        check_for_nans(logdxdls)
        check_for_nans(logd2xdl2s)
        return logdxdls, logd2xdl2s