import abc
from typing import Tuple

import torch

from ..domains import Domain


class Reference(abc.ABC):
    """Parent class for all one-dimensional reference distributions."""

    def __init__(self, domain: Domain):
        self.domain = domain

    @abc.abstractmethod
    def eval_cdf(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the CDF and gradient of the CDF (i.e., the PDF) of 
        the reference distribution at a set of values.
        
        Parameters
        ----------
        x:
            Set of values at which to evaluate the CDF and PDF of the 
            reference distribution.
            
        Returns
        -------
        :
            The CDF and PDF of the reference distribution evaluated at 
            each value of x.
        
        """
        return 
    
    @abc.abstractmethod
    def eval_pdf(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the PDF and gradient of the PDF of the reference
        distribution at each value of x.
        
        Parameters
        ----------
        x:
            Set of values at which to evaluate the PDF and gradient of 
            the PDF of the reference distribution.

        Returns
        -------
        :
            The PDF and gradient of the PDF of the reference 
            distribution evaluated at each element of x.
        
        """
        return
    
    @abc.abstractmethod
    def invert_cdf(
        self,
        u: torch.Tensor
    ) -> torch.Tensor:
        """Returns the values of the reference distribution 
        corresponding to a set of points on the CDF.
        
        Parameters
        ----------
        u: 
            A set of points on the CDF of the distribution.

        Returns
        -------
        : 
            The corresponding points of the distribution.
        
        """
        return
    
    @abc.abstractmethod
    def log_joint_pdf(
        self, 
        xs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the log-PDF and gradient of the log-PDF of each 
        elements of xs for the reference distribution.

        Parameters
        ----------
        xs:
            The vector at which to evaluate the log-PDF and gradient of 
            the log-PDF of the joint distribution. 

        Returns
        -------
        :
            The value of the log-PDF and gradient of the log-PDF of xs.

        """
        return
    
    @abc.abstractmethod
    def random(self, d: int, n: int) -> torch.Tensor:
        """Draws a set of samples from the reference distribution using
        the inverse CDF method.
        
        Parameters
        ----------
        d:
            The dimension of the samples.
        n:
            The number of samples to draw.

        Returns
        -------
        :
            The generated samples.

        """
        return