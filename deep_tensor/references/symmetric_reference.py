import abc
from typing import Tuple

import torch

from ..domains import BoundedDomain, Domain
from .reference import Reference


DEFAULT_MU = 0
DEFAULT_SIGMA = 1
DEFAULT_DOMAIN = BoundedDomain(bounds=torch.tensor([-4.0, 4.0]))
EPSILON = 1e-8


class SymmetricReference(Reference, abc.ABC):
    
    def __init__(
        self, 
        mu: float=DEFAULT_MU, 
        sigma: float=DEFAULT_SIGMA,
        domain: Domain=DEFAULT_DOMAIN
    ):
        
        self.mu = mu
        self.sigma = sigma
        self.domain = domain
        self.is_truncated = isinstance(domain, BoundedDomain)
        self.set_cdf_bounds()
        return

    def set_cdf_bounds(self):
        
        if self.is_truncated:
            self.left = self.eval_ref_cdf((self.domain.left-self.mu)/self.sigma)[0]
            self.right = self.eval_ref_cdf((self.domain.right-self.mu)/self.sigma)[0]
        else:
            self.left = 0.0
            self.right = 1.0
        return

    @abc.abstractmethod
    def eval_ref_cdf(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the values of the CDF and PDF of the unit reference
        distribution evaluated at each value of z.
        
        Parameters
        ----------
        z: 
            The values at which to evaluate the CDF and PDF of the 
            unit reference distribution.
        
        Returns
        -------
        :
            The CDF and PDF of the reference distribution evaluated at
            each value of z.
        
        """
        return
    
    @abc.abstractmethod
    def eval_ref_pdf(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the values of the PDF and gradient of the PDF of the 
        unit reference distribution evaluated at each value of z.
        
        Parameters
        ----------
        z: 
            The values at which to evaluate the PDF and gradient of the 
            PDF of the unit reference distribution.
        
        Returns
        -------
        :
            The PDF and gradient of the PDF of the reference 
            distribution evaluated at each value of z.
        
        """
        return

    @abc.abstractmethod
    def invert_ref_cdf(self, u: torch.Tensor) -> torch.Tensor:
        """Returns the inverse of the CDF of the unit reference 
        distribution evaluated at each value of u.
        
        Parameters
        ----------
        u:
            The values at which to evaluate the inverse CDF of the unit
            reference distribution.

        Returns
        -------
        :
            The inverse CDF of the reference distribution evaluated at
            each value of u.
        
        """
        return
    
    @abc.abstractmethod
    def log_joint_unit_pdf(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the log-PDF and gradient of the log-PDF of the 
        reference distribution evaluated at z.

        Parameters
        ----------
        z:
            An n-dimensional vector at which to evaluate the log-PDF 
            and gradient of the log-PDF of the reference distribution.

        Returns
        -------
        :
            The value of the log-PDF and gradient of the log-PDF of the 
            reference distribution at z.
        
        """
        return

    def map_to_unit(self, x: torch.Tensor) -> torch.Tensor:
        return (x-self.mu) / self.sigma

    def eval_cdf(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: might need to do some epsilon stuff in here.. 

        rs = self.map_to_unit(xs)
        zs, dzdrs = self.eval_ref_cdf(rs)

        zs = (zs - self.left) / (self.right - self.left)
        dzdrs = dzdrs / self.sigma / (self.right - self.left)
        
        return zs, dzdrs

    def invert_cdf(self, zs: torch.Tensor) -> torch.Tensor:

        zs[torch.isinf(zs)] = 1.0 - EPSILON
        zs[torch.isnan(zs)] = EPSILON

        # Map points into desired section of range of CDF
        zs = self.left + zs * (self.right-self.left)
        rs = self.invert_ref_cdf(zs)

        # Rescale points
        xs = self.mu + self.sigma * rs
        return xs
        
    def eval_pdf(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        z = self.map_to_unit(x)
        f, g = self.eval_ref_pdf(z)
        
        # TODO: figure out what's going on here.
        f = f / self.sigma / (self.right - self.left)
        g = g / self.sigma**2 / (self.right - self.left)

        return f, g
        
    def log_joint_pdf(
        self, 
        xs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        us = self.map_to_unit(xs)
        log_fus, log_gus = self.log_joint_unit_pdf(us)  # TODO: I think this (and some of the other functions) would be better as unit.

        log_fxs = log_fus - xs.shape[1] * (self.sigma * (self.right - self.left)).log()
        log_gxs = log_gus / self.sigma
    
        return log_fxs, log_gxs
    
    def random(self, d: int, n: int) -> torch.Tensor:

        zs = torch.rand(size=(n, d))
        zs = self.left + (self.right - self.left) * zs
        rs = self.invert_cdf(zs)
        return rs
        
    def sobol(self, d: int, n: int) -> torch.Tensor:

        S = torch.quasirandom.SobolEngine(dimension=d)
        zs = S.draw(n)
        zs = self.left + (self.right - self.left) * zs
        rs = self.invert_cdf(zs)
        return rs