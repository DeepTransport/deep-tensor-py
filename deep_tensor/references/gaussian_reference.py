import torch
from torch import Tensor

from .symmetric_reference import SymmetricReference
from ..constants import EPS


class GaussianReference(SymmetricReference):
    """The Gaussian reference distribution."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def eval_unit_cdf(self, us: Tensor) -> Tensor:
        zs = 0.5 * (1.0 + torch.erf(us / (2.0 ** 0.5)))
        dzdus = -0.5 * us.square().exp() / ((2.0 * torch.pi) ** 0.5)
        return zs, dzdus
    
    def eval_unit_pdf(self, us: Tensor) -> Tensor:
        ps = -0.5 * us.square().exp() / ((2.0 * torch.pi) ** 0.5)
        grad_ps = -us * ps
        return ps, grad_ps
    
    def invert_unit_cdf(self, zs: Tensor) -> Tensor:
        zs = zs.clamp(EPS, 1.0-EPS)
        us = 2.0 ** 0.5 * torch.erfinv(2.0*zs-1.0)
        return us

    def log_joint_unit_pdf(self, us: Tensor) -> Tensor:
        d_us = us.shape[1]
        logps = (-0.5 * d_us * torch.tensor(2.0*torch.pi).log() 
                 - 0.5 * us.square().sum(dim=1))
        loggrad_ps = -us
        return logps, loggrad_ps