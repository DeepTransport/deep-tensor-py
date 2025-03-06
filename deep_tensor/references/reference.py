import abc
from typing import Tuple

from torch import Tensor

from ..constants import EPS
from ..domains import Domain


class Reference(abc.ABC):
    """Parent class for all one-dimensional reference distributions."""

    def __init__(self, domain: Domain):
        self.domain = domain

    @abc.abstractmethod
    def eval_cdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the CDF and PDF (i.e., the gradient of the CDF) of 
        the reference distribution at a set of values.
        
        Parameters
        ----------
        rs:
            A matrix or vector containing a samples from the reference 
            density.
            
        Returns
        -------
        zs:
            A matrix or vector of the same dimension as rs, containing 
            the CDF of the reference density evaluated at each element 
            of rs.
        dzdrs:
            A matrix or vector of the same dimension as rs, containing 
            the PDF of the reference density evaluated at each element 
            of rs.
        
        """
        return 
    
    @abc.abstractmethod
    def eval_pdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the PDF and gradient of the PDF of the reference 
        distribution at a set of values.
        
        Parameters
        ----------
        rs:
            A matrix or vector containing a samples from the reference 
            density.
            
        Returns
        -------
        pdfs:
            A matrix or vector of the same dimension as rs, containing 
            the PDF of the reference density evaluated at each element 
            of rs.
        grad_pdfs:
            A matrix or vector of the same dimension as rs, containing 
            the gradient of the PDF of the reference density evaluated 
            at each element of rs.
        
        """
        return
    
    @abc.abstractmethod
    def invert_cdf(self, zs: Tensor) -> Tensor:
        """Returns the values of the reference distribution 
        corresponding to a set of points on the CDF.
        
        Parameters
        ----------
        zs: 
            A matrix or vector containing points distributed according 
            to the CDF of the distribution.

        Returns
        -------
        rs:
            A matrix or vector of the same dimension as zs, containing 
            the points from the reference density corresponding to each 
            element of zs.
        
        """
        return
    
    @abc.abstractmethod
    def log_joint_pdf(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the joint log-PDF and gradient of the log-PDF of 
        each of a set of points distributed according to the joint 
        reference density. 

        Parameters
        ----------
        rs:
            An n * d matrix containing samples for which to evaluate 
            the log-PDF and gradient of the log-PDF of the joint 
            reference density.

        Returns
        -------
        log_pdfs:
            A d-dimensional vector containing the log of the joint 
            reference density evaluated at each sample in rs.
        log_grad_pdfs:
            An n * d matrix containing the log of the gradient of the 
            joint reference density evaluated at each sample in rs.

        """
        return
    
    @abc.abstractmethod
    def random(self, d: int, n: int) -> Tensor:
        """Draws a set of samples from the reference density using
        the inverse CDF method.
        
        Parameters
        ----------
        d:
            The dimension of the samples.
        n:
            The number of samples to draw.

        Returns
        -------
        rs:
            An n * d matrix containing the generated samples.

        """
        return
    
    @abc.abstractmethod
    def sobol(self, d: int, n: int) -> Tensor:
        """Generates a set of QMC samples from the reference density 
        using a Sobol sequence.
        
        Parameters
        ----------
        d: 
            The dimension of the samples.
        n:
            The number of samples to generate.

        Returns
        -------
        rs:
            An n * d matrix containing the generated samples.
        
        """
        return
    
    def _check_samples_in_domain(self, rs: Tensor):
        """Raises an error if any of a set of samples are outside the
        domain of the reference.
        """
        outside = (rs < self.domain.left-EPS) & (self.domain.right+EPS < rs)
        if (n_outside := outside.sum()) > 0:
            msg = f"{n_outside} points lie outside domain of reference."
            raise Exception(msg)
        return