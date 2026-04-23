import abc
import logging
from typing import Tuple

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from ..domains import Domain


logger = logging.getLogger(__name__)


class Reference(abc.ABC):
    """Parent class for all one-dimensional reference distributions."""

    def __init__(self, domain: Domain):
        self.domain = domain
        return

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
        pass 
    
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
        pass
    
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
        pass
    
    @abc.abstractmethod
    def eval_potential(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the potential function (i.e., negative logarithm) 
        and the gradient of the potential function of the reference at 
        a set of points.

        Parameters
        ----------
        rs:
            An n * d matrix containing points at which to evaluate the 
            potential function and its gradient.

        Returns
        -------
        neglogrefs:
            An n-dimensional vector containing the potential function 
            evaluated at each sample in rs.
        grad_neglogrefs:
            An n * d matrix where each row contains the gradient of the 
            potential function evaluated at the corresponding sample in 
            rs.
        
        """
        pass

    @abc.abstractmethod
    def eval_potential_unnormalised(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the unnormalised potential function and the 
        gradient of the potential function of the reference at 
        a set of points. This can be useful for numerical stability.

        Parameters
        ----------
        rs:
            An n * d matrix containing points at which to evaluate the 
            potential function and its gradient.

        Returns
        -------
        neglogrefs:
            An n-dimensional vector containing the potential function 
            evaluated at each sample in rs.
        grad_neglogrefs:
            An n * d matrix where each row contains the gradient of the 
            potential function evaluated at the corresponding sample in 
            rs.
        
        """
        pass

    def _out_domain(self, rs: Tensor) -> Tensor:
        outside = (rs < self.domain.left) | (self.domain.right < rs)
        return outside
    
    def _check_samples_in_domain(self, rs: Tensor) -> None:
        """Raises a warning if any of a set of samples are outside the
        domain of the reference.

        TODO: check whether this is still used..
        """
        outside = self._out_domain(rs)
        if (num_outside := outside.sum()) > 0:
            msg = (
                f"{num_outside} points lie outside the domain of the "
                "reference distribution."
            )
            logger.debug(msg)
        return
    
    def _project_to_domain(self, rs: Tensor) -> Tensor:
        """Projects a set of samples to the nearest point in the 
        domain.
        """
        outside = self._out_domain(rs)
        if (num_outside := outside.sum()) > 0:
            msg = (
                f"{num_outside} points lie outside the domain of the "
                "reference distribution. Projecting each to the "
                "closest point in the domain of the reference "
                "distribution."
            )
            logger.debug(msg)
            rs = torch.clamp(rs, min=self.domain.left, max=self.domain.right)
        return rs
    
    def random(
        self, 
        n: int, 
        d: int, 
        device: torch.device = torch.get_default_device()
    ) -> Tensor:
        r"""Generates a set of random samples.
        
        Parameters
        ----------
        n:
            The number of samples to draw.
        d:
            The dimension of the samples.

        Returns
        -------
        rs:
            An $n \times d$ matrix containing the generated samples.

        """
        zs = torch.rand(n, d, device=device)
        rs = self.invert_cdf(zs)
        return rs
        
    def sobol(
        self, 
        n: int, 
        d: int,
        device: torch.device = torch.get_default_device()
    ) -> Tensor:
        r"""Generates a set of QMC samples.
        
        Parameters
        ----------
        n:
            The number of samples to generate.
        d: 
            The dimension of the samples.

        Returns
        -------
        rs:
            An $n \times d$ matrix containing the generated samples.
        
        """
        S = SobolEngine(dimension=d)
        zs = S.draw(n).to(device)
        rs = self.invert_cdf(zs)
        return rs