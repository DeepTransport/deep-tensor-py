from typing import Tuple

import torch
from torch import Tensor

from .symmetric_reference import SymmetricReference
from ..constants import EPS


class GaussianReference(SymmetricReference):
    r"""The standard $d$-dimensional Gaussian density, $\mathcal{N}(\boldsymbol{0}_{d}, \boldsymbol{I}_{d})$.

    The density can be truncated to a subinterval of the real numbers 
    in each dimension.
    
    Parameters
    ----------
    domain:
        The domain on which the density is defined in each dimension.
    
    """
    
    def eval_unit_cdf(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        zs = 0.5 * (1.0 + torch.erf(us / (2.0 ** 0.5)))
        dzdus = torch.exp(-0.5 * us.square()) / ((2.0 * torch.pi) ** 0.5)
        return zs, dzdus
    
    def eval_unit_pdf(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        ps = torch.exp(-0.5 * us.square()) / ((2.0 * torch.pi) ** 0.5)
        grad_ps = -us * ps
        return ps, grad_ps
    
    def invert_unit_cdf(self, zs: Tensor) -> Tensor:
        zs = zs.clamp(EPS, 1.0-EPS)
        us = 2.0 ** 0.5 * torch.erfinv(2.0*zs-1.0)
        return us

    def eval_unit_potential(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        d_us = us.shape[1]
        neglogps = (0.5 * d_us * torch.tensor(2.0*torch.pi).log() 
                 + 0.5 * us.square().sum(dim=1))
        grad_neglogps = us  # gradient of negative log
        return neglogps, grad_neglogps