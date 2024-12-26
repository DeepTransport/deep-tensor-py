import abc
from typing import Tuple

import torch

from .reference import Reference
from ..constants import EPS
from ..domains import BoundedDomain, Domain


class SymmetricReference(Reference, abc.ABC):
    
    def __init__(
        self, 
        mu: float=0.0, 
        sigma: float=1.0,
        domain: Domain=None
    ):
        
        if domain is None:
            bounds = torch.tensor([-4.0, 4.0])
            domain = BoundedDomain(bounds=bounds)

        self.mu = mu
        self.sigma = sigma
        self.domain = domain
        self.is_truncated = isinstance(domain, BoundedDomain)
        self.set_cdf_bounds()
        return

    @abc.abstractmethod
    def eval_unit_cdf(
        self, 
        us: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the values of the CDF and PDF of the unit reference
        distribution evaluated at each value of us.
        
        Parameters
        ----------
        us: 
            A vector or matrix of values at which to evaluate the 
            CDF and PDF of the unit reference distribution.
        
        Returns
        -------
        zs:
            A vector or matrix of the same dimension as us, containing 
            the CDF of the unit reference distribution evaluated at 
            each element of us.
        dzdus:
            A vector or matrix of the same dimension as us, containing 
            the PDF of the unit reference distribution evaluated at 
            each element of us.
        
        """
        return
    
    @abc.abstractmethod
    def eval_unit_pdf(
        self, 
        us: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the values of the PDF and gradient of the PDF of the 
        unit reference distribution evaluated at each value of us.
        
        Parameters
        ----------
        us: 
            A vector or matrix of values at which to evaluate the 
            PDF and gradient of the PDF of the unit reference 
            distribution.
        
        Returns
        -------
        pdfs:
            A vector or matrix of the same dimension as us, containing 
            the PDF of the unit reference distribution evaluated at 
            each element of us.
        grad_pdfs:
            A vector or matrix of the same dimension as us, containing 
            the gradient of the PDF of the unit reference distribution 
            evaluated at each element of us.
        
        """
        return

    @abc.abstractmethod
    def invert_unit_cdf(self, zs: torch.Tensor) -> torch.Tensor:
        """Returns the inverse of the CDF of the unit reference 
        distribution evaluated at each element of zs.
        
        Parameters
        ----------
        zs:
            A matrix or vector containg values at which to evaluate the 
            inverse of the CDF of the unit reference distribution.

        Returns
        -------
        us:
            A matrix or vector of the same dimension as zs, containing 
            the inverse of the CDF of the unit reference distribution 
            evaluated at each element of zs.
        
        """
        return
    
    @abc.abstractmethod
    def log_joint_unit_pdf(
        self, 
        us: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the log-PDF and gradient of the log-PDF of the 
        reference distribution evaluated at each element of us.

        Parameters
        ----------
        us:
            An n * d matrix vector containing samples distributed 
            according to the joint reference distribution.

        Returns
        -------
        logpdfs:
            A d-dimensional vector containing the PDF of the joint unit 
            reference distribution evaluated at each sample in us.
        loggrad_pdfs:
            An n * d matrix containing the log of the gradient of the 
            joint unit reference density evaluated at each sample in 
            us.
        
        """
        return
    
    def set_cdf_bounds(self):
        """Sets the minimum and maximum possible values of the CDF 
        based on the bounds of the domain.
        """
        
        if self.is_truncated:
            sigma_left = (self.domain.left-self.mu) / self.sigma
            sigma_right = (self.domain.right-self.mu) / self.sigma
            self.left = self.eval_unit_cdf(sigma_left)[0]
            self.right = self.eval_unit_cdf(sigma_right)[0]
        else:
            self.left = 0.0
            self.right = 1.0

        # Normalising constant for PDF
        self.norm = self.right - self.left

        return

    def map_to_unit(self, xs: torch.Tensor) -> torch.Tensor:
        """Maps a set of variates from the reference density to samples
        from the density of the same form, but with zero mean and unit 
        variance.
        
        Parameters
        ----------
        rs:
            An n * d matrix containing samples from the reference 
            density.
        
        Returns
        -------
        us:
            The corresponding samples after transforming such that they
            have zero mean and unit variance.
        
        """

        us = (xs - self.mu) / self.sigma
        return us

    def eval_cdf(
        self, 
        rs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        us = self.map_to_unit(rs)
        zs, dzdus = self.eval_unit_cdf(us)

        zs = (zs - self.left) / self.norm
        dzdus /= (self.sigma * self.norm)
        
        return zs, dzdus
    
    def eval_pdf(
        self, 
        rs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        us = self.map_to_unit(rs)
        pdfs, grad_pdfs = self.eval_unit_pdf(us)
        
        pdfs /= (self.sigma * self.norm)
        grad_pdfs /= (self.sigma**2 * self.norm)

        return pdfs, grad_pdfs

    def invert_cdf(self, zs: torch.Tensor) -> torch.Tensor:

        zs[torch.isinf(zs)] = 1.0 - EPS
        zs[torch.isnan(zs)] = EPS

        # Map points into desired section of range of CDF
        zs = self.left + zs * self.norm
        us = self.invert_unit_cdf(zs)
        rs = self.mu + self.sigma * us
        return rs
        
    def log_joint_pdf(
        self, 
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        dim_x = xs.shape[1]

        us = self.map_to_unit(xs)
        log_pdfs, log_grad_pdfs = self.log_joint_unit_pdf(us)

        log_pdfs -= dim_x * (self.sigma * self.norm).log()
        log_grad_pdfs /= self.sigma
    
        return log_pdfs, log_grad_pdfs
    
    def random(self, d: int, n: int) -> torch.Tensor:

        zs = torch.rand(size=(n, d))
        zs = self.left + self.norm * zs
        rs = self.invert_cdf(zs)
        return rs
        
    def sobol(self, d: int, n: int) -> torch.Tensor:

        S = torch.quasirandom.SobolEngine(dimension=d)
        zs = S.draw(n)
        zs = self.left + self.norm * zs
        rs = self.invert_cdf(zs)
        return rs