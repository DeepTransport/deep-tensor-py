from typing import Tuple

import torch 
from torch import Tensor 

from .laguerre import Laguerre
from .spectral_cdf import SpectralCDF
from ...constants import EPS


class LaguerreCDF(Laguerre, SpectralCDF):

    def __init__(self, poly: Laguerre, **kwargs):
        Laguerre.__init__(self, 2*poly.order)
        SpectralCDF.__init__(**kwargs)
    
    def grid_measure(self, n: int) -> Tensor:
        b = torch.maximum(self.nodes[-1], torch.tensor(15.))
        ls = torch.linspace(EPS, b, n)
        return ls 
    
    def eval_int_basis(self, ls: Tensor) -> Tensor:
        ps = self.eval_basis(ls)
        ws = self.eval_measure(ls)
        ts = ps[:, :-1].cumsum(dim=1)
        ps[:, 0] = -ws
        ps[:, 1:] = (ts * (ws * ls)[:, None]) / torch.arange(1, self.order+1)
        return ps
    
    def eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        ps = self.eval_basis(ls)
        ws = self.eval_measure(ls)
        dpdls = ps * ws[:, None]
        ts = ps[:, :-1].cumsum(dim=1)
        ps[:, 0] = -ws[:, None]
        ps[:, 1:] = (ts * (ws * ls)[:, None]) / torch.arange(1, self.order+1)
        return ps, dpdls