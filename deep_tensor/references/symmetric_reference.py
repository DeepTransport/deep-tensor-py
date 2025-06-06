import abc
from typing import Tuple

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from .reference import Reference
from ..domains import BoundedDomain, Domain
from ..tools import check_finite


class SymmetricReference(Reference, abc.ABC):
    
    def __init__(self, domain: Domain | None = None):
        
        if domain is None:
            bounds = torch.tensor([-4.0, 4.0])
            domain = BoundedDomain(bounds=bounds)

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
        dpdus:
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
    def eval_unit_potential(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the negative log-PDF and gradient of the negative 
        log-PDF of the reference distribution evaluated at each element 
        of us.

        Parameters
        ----------
        us:
            An n * d matrix vector containing samples distributed 
            according to the joint reference distribution.

        Returns
        -------
        log_ps:
            A d-dimensional vector containing the PDF of the joint unit 
            reference distribution evaluated at each sample in us.
        log_dpdus:
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
            self.cdf_left = self.eval_unit_cdf(self.domain.left)[0]
            self.cdf_right = self.eval_unit_cdf(self.domain.right)[0]
        else:
            self.cdf_left = 0.0
            self.cdf_right = 1.0

        # Normalising constant for PDF
        self.norm = self.cdf_right - self.cdf_left
        return

    def eval_cdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_samples_in_domain(rs)
        zs, dzdrs = self.eval_unit_cdf(rs)
        zs = (zs - self.cdf_left) / self.norm
        dzdrs = dzdrs / self.norm
        return zs, dzdrs
    
    def eval_pdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_samples_in_domain(rs)
        ps, dpdrs = self.eval_unit_pdf(rs)
        ps = ps / self.norm
        dpdrs = dpdrs / self.norm
        return ps, dpdrs

    def invert_cdf(self, zs: Tensor) -> Tensor:
        check_finite(zs)
        zs = self.cdf_left + zs * self.norm
        us = self.invert_unit_cdf(zs)
        return us
        
    def eval_potential(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        self._check_samples_in_domain(rs)
        d_rs = rs.shape[1]
        log_ps, log_dpdrs = self.eval_unit_potential(rs)
        log_ps = log_ps + d_rs * self.norm.log()
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