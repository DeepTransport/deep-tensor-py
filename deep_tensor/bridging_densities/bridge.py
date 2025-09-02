import abc
from typing import List, Tuple

import torch
from torch import Tensor

from ..preconditioners import Preconditioner
from ..target_functions import TargetFunc


class Bridge(abc.ABC):
    
    @property
    @abc.abstractmethod
    def is_last(self) -> bool:
        pass

    @property 
    def n_layers(self) -> int:
        return self._n_layers
    
    @n_layers.setter
    def n_layers(self, value: int) -> None:
        self._n_layers = value
        return
    
    @property
    def is_adaptive(self) -> bool:
        return self._is_adaptive
    
    @is_adaptive.setter 
    def is_adaptive(self, value: bool) -> None:
        self._is_adaptive = value 
        return

    @abc.abstractmethod
    def ratio_func(
        self, 
        method: str,
        rs: Tensor,
        us: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        pass

    @abc.abstractmethod
    def update(self, us: Tensor, neglogfus_dirt: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the current bridging density, the next ratio 
        function and the ratio between the current bridging density and 
        the next bridging density at each of a set of samples.

        Parameters
        ----------
        us:
            An n * d matrix containing the samples from `rs` after 
            applying the current DIRT mapping to them.
        neglogfus_dirt:
            An n-dimensional vector containing evaluations of the 
            current DIRT density at each sample in us.  
        
        Returns
        -------
        log_weights:
            An n-dimensional vector containing the ratio between 
            the current and new bridging densities evaluated at each 
            sample.
        neglogbridges:
            An n-dimensional vector containing the current bridging 
            density evaluated at each sample.
        
        """
        pass

    def initialise(
        self, 
        preconditioner: Preconditioner, 
        target_func: TargetFunc
    ) -> None:
        self.preconditioner = preconditioner
        self.reference = self.preconditioner.reference
        self.target_func = target_func
        return
    
    def apply_preconditioner(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        xs = self.preconditioner.Q(us)
        neglogdets = self.preconditioner.neglogdet_Q(us)
        return xs, neglogdets
    
    def _reorder(
        self, 
        xs: Tensor, 
        neglogratios: Tensor,
        log_weights: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Returns a reordered set of indices based on the importance
        weights between the current bridging density and the density 
        of the approximation to the previous target density evaluated
        at a set of samples from the approximation to the previous 
        target density.

        Parameters
        ----------
        xs:
            An n * d matrix containing a set of samples distributed 
            according to the approximation to the previous target 
            density.
        neglogratios:
            An n-dimensional vector containing the negative logarithm 
            of the ratio function evaluated at each sample in xs.
        log_weights:
            An n-dimensional vector containing the logarithm of the 
            ratio between the current bridging density and the density 
            of the approximation to the previous target density 
            evaluated at each sample in xs.

        Returns
        -------
        xs:
            An n * d matrix containing the reordered samples.
        neglogratios:
            An n-dimensional vector containing the negative logarithm 
            of the ratio function evaluated at each sample in xs.

        """
        reordered_inds = torch.argsort(log_weights).flip(dims=(0,))
        xs = xs[reordered_inds]
        neglogratios = neglogratios[reordered_inds]
        return xs, neglogratios

    def _get_diagnostics(
        self, 
        log_weights: Tensor,
        neglogfus: Tensor,
        neglogfus_dirt: Tensor
    ) -> List[str]:
        """Returns some information about the current bridging density.

        Parameters
        ----------
        log_weights:
            An n-dimensional vector containing the logarithm of the 
            ratio between the next bridging density and the current 
            bridging density evaluated at a set of samples from the 
            DIRT approximation to the current bridging density.
        neglogfus:
            An n-dimensional vector containing the negative logarithm 
            of the current bridging density evaluated at a set of 
            samples from the DIRT approximation.
        neglogfus_dirt:
            An-dimensional vector containin the negative logarithm of 
            the DIRT approximation to the current bridging density 
            evaluated at the same set of samples as above. 

        """
        return []