import abc
from typing import Tuple

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from .reference import Reference
from ..domains import BoundedDomain, Domain
from ..tools import check_finite


class SymmetricReference(Reference, abc.ABC):
    
    def __init__(
        self, 
        mu: float = 0.0, 
        sigma: float = 1.0,
        domain: Domain = None
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
    def eval_unit_cdf(self, us: Tensor) -> Tuple[Tensor, Tensor]:
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
    def eval_unit_pdf(self, us: Tensor) -> Tuple[Tensor, Tensor]:
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
        ps:
            A vector or matrix of the same dimension as us, containing 
            the PDF of the unit reference distribution evaluated at 
            each element of us.
        grad_ps:
            A vector or matrix of the same dimension as us, containing 
            the gradient of the PDF of the unit reference distribution 
            evaluated at each element of us.
        
        """
        return

    @abc.abstractmethod
    def invert_unit_cdf(self, zs: Tensor) -> Tensor:
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
    def log_joint_unit_pdf(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the log-PDF and gradient of the log-PDF of the 
        reference distribution evaluated at each element of us.

        Parameters
        ----------
        us:
            An n * d matrix vector containing samples distributed 
            according to the joint reference distribution.

        Returns
        -------
        logps:
            A d-dimensional vector containing the PDF of the joint unit 
            reference distribution evaluated at each sample in us.
        loggrad_ps:
            An n * d matrix containing the log of the gradient of the 
            joint unit reference density evaluated at each sample in 
            us.
        
        """
        return
    
    def set_cdf_bounds(self) -> None:
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

    def map_to_unit(self, xs: Tensor) -> Tensor:
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

    def eval_cdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_samples_in_domain(rs)
        us = self.map_to_unit(rs)
        zs, dzdus = self.eval_unit_cdf(us)
        zs = (zs - self.left) / self.norm
        dzdrs = dzdus / (self.sigma * self.norm)
        return zs, dzdrs
    
    def eval_pdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_samples_in_domain(rs)
        us = self.map_to_unit(rs)
        ps, grad_ps = self.eval_unit_pdf(us)
        ps /= (self.sigma * self.norm)
        grad_ps /= (self.sigma**2 * self.norm)
        return ps, grad_ps

    def invert_cdf(self, zs: Tensor) -> Tensor:
        check_finite(zs)
        zs = self.left + zs * self.norm
        us = self.invert_unit_cdf(zs)
        rs = self.mu + self.sigma * us
        return rs
        
    def log_joint_pdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_samples_in_domain(rs)
        d_rs = rs.shape[1]
        us = self.map_to_unit(rs)
        log_ps, log_grad_ps = self.log_joint_unit_pdf(us)
        log_ps -= d_rs * (self.sigma * self.norm).log()
        log_grad_ps /= self.sigma
        return log_ps, log_grad_ps
    
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
        zs = torch.rand(n, d)
        rs = self.invert_cdf(zs)
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
        zs = S.draw(n)
        rs = self.invert_cdf(zs)
        return rs