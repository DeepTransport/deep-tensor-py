from typing import Tuple

import torch
from torch import Tensor 
from torch.quasirandom import SobolEngine

from .reference import Reference
from ..domains import BoundedDomain


class UniformReference(Reference):
    r"""The standard $d$-dimensional uniform density, $\mathcal{U}([0, 1]^{d})$.
    """

    def __init__(self):
        self.domain = BoundedDomain(bounds=torch.tensor([0.0, 1.0]))
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
        ps = torch.ones_like(rs)
        dpdrs = torch.zeros_like(rs)
        return ps, dpdrs
    
    def eval_potential(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_samples_in_domain(rs)
        n_rs = rs.shape[0]
        log_ps = torch.zeros(n_rs)
        log_dpdrs = torch.zeros(n_rs)  # TODO: look at this more closely.
        return log_ps, log_dpdrs
    
    def random(self, d: int, n: int) -> Tensor:
        r"""Generates a set of random samples.
        
        Parameters
        ----------
        d:
            The dimension of the samples.
        n:
            The number of samples to draw.

        Returns
        -------
        rs:
            An $n \times d$ matrix containing the generated samples.

        """
        rs = torch.rand(n, d)
        return rs
    
    def sobol(self, d: int, n: int) -> Tensor:
        r"""Generates a set of QMC samples.
        
        Parameters
        ----------
        d: 
            The dimension of the samples.
        n:
            The number of samples to generate.

        Returns
        -------
        rs:
            An $n \times d$ matrix containing the generated samples.
        
        """
        S = SobolEngine(dimension=d)
        rs = S.draw(n)
        return rs