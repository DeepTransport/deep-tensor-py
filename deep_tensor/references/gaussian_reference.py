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
        dzdus = torch.exp(-0.5 * us**2) / ((2.0 * torch.pi) ** 0.5)
        return zs, dzdus
    
    def eval_unit_pdf(self, us: Tensor) -> Tensor:
        pdfs = torch.exp(-0.5 * us**2) / ((2.0 * torch.pi) ** 0.5)
        grad_pdfs = -us * pdfs
        return pdfs, grad_pdfs
    
    def invert_unit_cdf(self, zs: Tensor) -> Tensor:
        zs = zs.clamp(EPS, 1.0-EPS)
        us = (2.0 ** 0.5) * torch.erfinv(2.0*zs-1.0)
        return us

    def log_joint_unit_pdf(self, us: Tensor) -> Tensor:
        dim_u = us.shape[1]
        logpdfs = (-0.5 * dim_u * torch.log(torch.tensor(2.0*torch.pi)) 
                   + torch.sum(-0.5 * us**2, dim=1))
        loggrad_pdfs = -us
        return logpdfs, loggrad_pdfs