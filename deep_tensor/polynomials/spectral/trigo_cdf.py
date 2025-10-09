import abc

import torch
from torch import Tensor

from .spectral import Spectral
from .spectral_cdf import SpectralCDF


class TrigoCDF(SpectralCDF, abc.ABC):

    def __init__(self, error_tol: float, device: torch.device):
        SpectralCDF.__init__(self, error_tol=error_tol, device=device)
        return
    
    def grid_measure(self, n: int) -> Tensor:
        return torch.linspace(-torch.pi, 0.0, n, device=self.device)
    
    def eval_int_deriv(self, ps: Tensor, ls: Tensor) -> Tensor:
        thetas = Spectral.l2theta(ls)
        zs = SpectralCDF.eval_int_deriv(self, ps, -thetas)
        return zs
    
    def eval_cdf(self, ps: Tensor, ls: Tensor) -> Tensor:
        thetas = Spectral.l2theta(ls)
        zs = SpectralCDF.eval_cdf(self, ps, -thetas)
        return zs 
    
    def invert_cdf(self, ps: Tensor, zs: Tensor):
        ls = SpectralCDF.invert_cdf(self, ps, zs)
        return torch.cos(-ls)