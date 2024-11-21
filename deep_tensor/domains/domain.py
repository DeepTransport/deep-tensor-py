import abc
from typing import Tuple

import torch


class Domain(abc.ABC):

    @property
    @abc.abstractmethod
    def bounds(self) -> torch.Tensor:
        return
    
    @abc.abstractmethod
    def reference2domain(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps a set of points in the reference domain to the 
        approximation domain.
        
        Parameters
        ----------
        z: 
            The set of points in the reference domain.

        Returns
        -------
        :
            The corresponding points in the approximation domain, and 
            the gradient of the mapping from the reference domain to 
            the approximation domain evaluated at each point.
            
        """
        return
    
    @abc.abstractmethod
    def domain2reference(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps a set of points in the approximation domain back to the 
        reference domain.
        
        Parameters
        ----------
        x:
            Set of points in the approximation domain.

        Returns
        -------
        :
            The corresponding points in the reference domain, and the 
            gradient of the mapping from the reference domain to the 
            approximation domain evaluated at each point.

        """
        return
    
    @abc.abstractmethod
    def reference2domain_log_density(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: write this
        """
        return
    
    @abc.abstractmethod
    def domain2reference_log_density(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: write this
        """
        return