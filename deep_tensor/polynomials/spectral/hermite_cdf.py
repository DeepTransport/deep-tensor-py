from typing import Tuple

import torch
from torch import Tensor

from .hermite import Hermite
from .spectral_cdf import SpectralCDF


class HermiteCDF(Hermite, SpectralCDF):

    def __init__(self, poly: Hermite, **kwargs):
        Hermite.__init__(self, 2*poly.order)
        SpectralCDF.__init__(self, **kwargs)
        return
    
    def grid_measure(self, n: int):
        # TODO: think about this.
        ls = torch.linspace(-10., 10., n)
        return ls
    
    def eval_int_basis(self, ls: Tensor) -> Tensor:
        ps = self.eval_basis(ls)
        ws = self.eval_measure(ls)[:, None]
        ps[:, 1:] = -ps[:, :-1] * self.norm[1:] / self.norm[:-1]
        ps *= ws
        ps[:, 0] = 0.5 * torch.erf(ls / torch.tensor(2.0).sqrt())
        return ps
    
    def eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        ps = self.eval_basis(ls)
        ws = self.eval_measure(ls)[:, None]
        dpdls = ps * ws
        ps[:, 1:] = -ps[:, :-1] * self.norm[1:] / self.norm[:-1]
        ps *= ws
        ps[:, 0] = 0.5 * torch.erf(ls / torch.tensor(2.0).sqrt())
        return ps, dpdls