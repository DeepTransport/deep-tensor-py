import abc
from typing import Tuple

import torch


class Domain(abc.ABC):
    """Parent class for all approximation domains."""

    @property
    @abc.abstractmethod
    def bounds(self) -> torch.Tensor:
        """The boundary of the approximation domain."""
        return
    
    @property
    @abc.abstractmethod 
    def left(self) -> torch.Tensor:
        """The left-hand boundary of the approximation domain."""
        return 
    
    @property 
    @abc.abstractmethod
    def right(self) -> torch.Tensor:
        """The right-hand boundary of the approximation domain."""
        return
    
    @abc.abstractmethod
    def reference2domain(
        self, 
        rs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps a set of points in the reference domain to the 
        approximation domain.
        
        Parameters
        ----------
        rs: 
            An n-dimensional vector containing points from the reference 
            domain.

        Returns
        -------
        xs:
            An n-dimensional vector containing the corresponding points
            in the approximation domain.
        dxdrs:
            An n-dimensional vector containing the gradient of the 
            mapping from the reference domain to the approximation 
            domain evaluated at each point in xs.
            
        """
        return
    
    @abc.abstractmethod
    def domain2reference(
        self, 
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps a set of points in the approximation domain back to the 
        reference domain.
        
        Parameters
        ----------
        xs:
            An n-dimensional vector containing points from the 
            approximation domain.

        Returns
        -------
        rs:
            An n-dimensional vector containing the corresponding points 
            in the reference domain.
        drdxs:
            An n-dimensional vector containing the gradient of the 
            mapping from the approximation domain to the reference 
            domain evaluated at each point in rs.

        """
        return
    
    @abc.abstractmethod
    def reference2domain_log_density(
        self, 
        rs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the logarithm of the derivative of the mapping from 
        the reference domain to the approximation domain and its
        gradient.

        Parameters
        ----------
        rs: 
            An n-dimensional vector containing a set of points from the 
            reference domain.
        
        Returns
        -------
        logdxdrs:
            An n-dimensional vector containing the logarithm of the 
            gradient of the mapping from the reference domain to the 
            approximation domain.
        logdxdr2s:
            An n-dimensional vector containing the logarithm of the 
            derivative of the gradient of the mapping from the 
            reference domain to the approximation domain.
        
        """
        return
    
    @abc.abstractmethod
    def domain2reference_log_density(
        self, 
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the logarithm of the derivative of the mapping from 
        the approximation domain to the reference domain and its
        gradient.

        Parameters
        ----------
        xs: 
            An n-dimensional vector containing a set of points from the 
            approximation domain.
        
        Returns
        -------
        logdrdxs:
            An n-dimensional vector containing the logarithm of the 
            gradient of the mapping from the approximation domain to 
            the reference domain.
        logdrdx2s:
            An n-dimensional vector containing the logarithm of the 
            derivative of the gradient of the mapping from the 
            approximation domain to the reference domain.
        
        """
        return