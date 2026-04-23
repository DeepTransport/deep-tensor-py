from typing import Tuple

import torch
from torch import Tensor 

from .reference import Reference
from ..domains import BoundedDomain


class UniformReference(Reference):
    r"""The standard $d$-dimensional uniform density, $\mathcal{U}([0, 1]^{d})$.
    """

    def __init__(self):
        self.domain = BoundedDomain([0.0, 1.0])
        self.pdf = 1.0
        return
    
    def invert_cdf(self, zs: Tensor) -> Tensor:
        return zs
    
    def eval_cdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_samples_in_domain(rs)
        zs = rs.clone()
        dzdrs = torch.ones_like(rs)
        return zs, dzdrs 
    
    def eval_pdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_samples_in_domain(rs)
        prs = torch.ones_like(rs)
        grad_prs = torch.zeros_like(rs)
        return prs, grad_prs
    
    def eval_potential(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_samples_in_domain(rs)
        neglogprs = torch.zeros((rs.shape[0],), device=rs.device)
        grad_neglogprs = torch.zeros_like(rs)
        return neglogprs, grad_neglogprs
    
    def eval_potential_unnormalised(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        return self.eval_potential(rs)