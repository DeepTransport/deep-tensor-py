from typing import Tuple

import torch
from torch import Tensor 

from .reference import Reference
from ..constants import EPS
from ..tools import check_for_nans


class UniformReference(Reference):

    def __init__(self, domain: Tensor|None = None):
        
        if domain is None:
            domain = torch.tensor([-1., 1.])
        
        self.domain = domain 
        self.pdf = 1. / (self.domain[1] - self.domain[0])
        return
    
    def invert_cdf(self, zs: Tensor) -> Tensor:
        check_for_nans(zs)
        zs = torch.clamp(zs, EPS, 1.-EPS)
        rs = self.domain[0] + zs / self.pdf
        return rs
    
    def eval_cdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        zs = self.pdf * (rs - self.domain[0])
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
        S = torch.quasirandom.SobolEngine(dimension=d)
        zs = S.draw(n)
        rs = self.invert_cdf(zs)
        return rs