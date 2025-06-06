import abc
from typing import Tuple

from torch import Tensor


class Domain(abc.ABC):
    """Parent class for all approximation domains."""

    @property
    @abc.abstractmethod
    def bounds(self) -> Tensor:
        """The boundary of the approximation domain."""
        return
    
    @property
    def left(self) -> Tensor:
        """The left-hand boundary of the approximation domain."""
        return self.bounds[0]
    
    @property 
    def right(self) -> Tensor:
        """The right-hand boundary of the approximation domain."""
        return self.bounds[1]

    @abc.abstractmethod
    def local2approx(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        """Maps a set of points in the local domain to the 
        approximation domain.
        
        Parameters
        ----------
        ls: 
            An n-dimensional vector containing points from the local 
            domain.

        Returns
        -------
        xs:
            An n-dimensional vector containing the corresponding points
            in the approximation domain.
        dxdls:
            An n-dimensional vector containing the gradient of the 
            mapping from the local domain to the approximation 
            domain evaluated at each point in xs.
            
        """
        return
    
    @abc.abstractmethod
    def approx2local(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        """Maps a set of points in the approximation domain back to the 
        local domain.
        
        Parameters
        ----------
        xs:
            An n-dimensional vector containing points from the 
            approximation domain.

        Returns
        -------
        ls:
            An n-dimensional vector containing the corresponding points 
            in the local domain.
        dldxs:
            An n-dimensional vector containing the gradient of the 
            mapping from the approximation domain to the local 
            domain evaluated at each point in rs.

        """
        return
    
    @abc.abstractmethod
    def local2approx_log_density(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the logarithm of the derivative of the mapping from 
        the local domain to the approximation domain and its
        gradient.

        Parameters
        ----------
        ls: 
            An n-dimensional vector containing a set of points from the 
            local domain.
        
        Returns
        -------
        logdxdls:
            An n-dimensional vector containing the logarithm of the 
            gradient of the mapping from the local domain to the 
            approximation domain.
        logdxdl2s:
            An n-dimensional vector containing the logarithm of the 
            second derivative of the mapping from the local domain to 
            the approximation domain.
        
        """
        return
    
    @abc.abstractmethod
    def approx2local_log_density(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the logarithm of the derivative of the mapping from 
        the approximation domain to the local domain and its
        gradient.

        Parameters
        ----------
        xs: 
            An n-dimensional vector containing a set of points from the 
            approximation domain.
        
        Returns
        -------
        logdldxs:
            An n-dimensional vector containing the logarithm of the 
            gradient of the mapping from the approximation domain to 
            the local domain.
        logdldx2s:
            An n-dimensional vector containing the derivative of the 
            logarithm of the gradient of the mapping from the 
            approximation domain to the local domain.
        
        """
        return
    
    @staticmethod
    def check_bounds(bounds: Tensor) -> None:
        if bounds[0] >= bounds[1]:
            msg = "Left-hand bound must be less than right-hand bound."
            raise Exception(msg)
        return