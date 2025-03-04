from typing import Tuple

import torch
from torch import Tensor 
from torch.quasirandom import SobolEngine

from .reference import Reference
from ..constants import EPS
from ..domains import BoundedDomain
from ..tools import check_finite


class UniformReference(Reference):
    """The uniform reference density.
    
    Parameters
    ----------
    domain:
        The domain on which the density is defined.
    
    """

    def __init__(self, domain: BoundedDomain|None = None):
        
        if domain is None:
            bounds = torch.tensor([-4.0, 4.0])
            domain = BoundedDomain(bounds=bounds)
        
        self.domain = domain 
        self.pdf = 1.0 / (self.domain.right - self.domain.left)
        return
    
    def invert_cdf(self, zs: Tensor) -> Tensor:
        check_finite(zs)
        zs = torch.clamp(zs, EPS, 1.0-EPS)
        rs = self.domain.left + zs / self.pdf
        return rs
    
    def eval_cdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        zs = self.pdf * (rs - self.domain.left)
        dzdrs = self.pdf * torch.ones_like(rs)
        return zs, dzdrs 
    
    def eval_pdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        pdfs = self.pdf * torch.ones_like(rs)
        grad_pdfs = torch.zeros_like(rs)
        return pdfs, grad_pdfs
    
    def log_joint_pdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        n_rs, d_rs = rs.shape
        log_pdfs = torch.full((n_rs,), self.pdf.log() * d_rs)
        log_grad_pdfs = torch.zeros(n_rs)
        return log_pdfs, log_grad_pdfs
    
    def random(self, d: int, n: int) -> Tensor:
        zs = torch.rand(n, d)
        rs = self.invert_cdf(zs)
        return rs 
    
    def sobol(self, d: int, n: int) -> Tensor:
        S = SobolEngine(dimension=d)
        zs = S.draw(n)
        rs = self.invert_cdf(zs)
        return rs