import abc
from typing import Dict, List, Tuple

import torch
from torch import Tensor


class Bridge(abc.ABC):

    @property 
    @abc.abstractmethod 
    def is_adaptive(self) -> bool:
        return
    
    @property
    @abc.abstractmethod
    def is_last(self) -> bool:
        return
    
    @property 
    @abc.abstractmethod
    def params_dict(self) -> Dict:
        return

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
    def _get_ratio_func(
        self, 
        method: str,
        rs: Tensor,
        neglogliks: Tensor, 
        neglogpris: Tensor, 
        neglogfxs: Tensor
    ) -> Tensor:
        """Returns the negative log-ratio function evaluated each of 
        the current set of samples.
        
        Parameters
        ----------
        method:
            The method used to compute the ratio function. Can be
            'eratio' (exact) or 'aratio' (approximate).
        rs:
            An n * d matrix containing a set of samples from the 
            reference density.
        neglogliks:
            An n-dimensional vector containing the negative 
            log-likelihood evaluated at each sample.
        neglogpris:
            An n-dimensional vector containing the negative log-prior
            density evaluated at each sample.
        neglogfxs:
            An n-dimensional vector containing the negative logarithm
            of the current DIRT density.

        Returns
        -------
        neglogratio:
            An n-dimensional vector containing the negative log-ratio 
            function evaluated for each sample.
            
        """
        return
    
    @abc.abstractmethod
    def _compute_log_weights(
        self,
        neglogliks: Tensor,
        neglogpris: Tensor, 
        neglogfxs: Tensor
    ) -> Tensor:
        """Returns the logarithm of the ratio between the current 
        bridging density and the density of the approximation to the 
        previous bridging density evaluated at each of a set of samples
        distributed according to the previous bridging density.

        Parameters
        ----------
        neglogliks:
            An n-dimensional vector containing the negative 
            log-likelihood function evaluated at each sample.
        neglogpris:
            An n-dimensional vector containing the negative log-prior 
            density evaluated at each sample.
        neglogfxs:
            An n-dimensional vector containing the negative logarithm 
            of the approximation to the previous bridging density 
            evaluated at each sample.

        Returns
        -------
        log_weights:
            The logarithm of the ratio between the current bridging 
            density and the density of the approximation to the 
            previous bridging density evaluated at each sample.
        
        """
        return
    
    def _set_init(self, neglogliks: Tensor) -> None:
        """Computes the properties of the initial bridging density.
        
        Parameters
        ----------
        neglogliks:
            An n-dimensional vector containing the negative 
            log-likelihood function evaluated at a set of 
            initialisation samples (drawn from the prior).

        Returns
        -------
        None
        
        """
        return 
    
    def _adapt_density(
        self,
        method: str, 
        neglogliks: Tensor, 
        neglogpris: Tensor, 
        neglogfxs: Tensor
    ) -> None:
        """Determines the beta value associated with the next bridging 
        density.
        
        Parameters
        ----------
        method: 
            The method used to select the next bridging parameter. Can
            be 'aratio' (approximate ratio) or 'eratio' (exact ratio).
        neglogliks: 
            An n-dimensional vector containing the negative 
            log-likelihood of each of the current samples.
        neglogpris:
            An n-dimensional vector containing the negative log-prior 
            density of each of the current samples.
        neglogfxs:
            An n-dimensional vector containing the negative log-density 
            of the current approximation to the target density for each 
            of the current samples.

        Returns
        -------
        None
        
        """
        return
    
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
        neglogweights: Tensor,
        neglogliks: Tensor, 
        neglogpris: Tensor, 
        neglogfxs: Tensor
    ) -> List[str]:
        """Returns some information about the current bridging density.
        """
        return []