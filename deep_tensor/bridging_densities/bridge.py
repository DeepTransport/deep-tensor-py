import abc
from typing import Callable

import torch

from ..references import Reference


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
    def num_layers(self) -> int:
        return

    @abc.abstractmethod
    def get_ratio_func(
        self, 
        reference: Reference, 
        method: str,
        xs: torch.Tensor,
        neglogliks: torch.Tensor, 
        neglogpris: torch.Tensor, 
        neglogfxs: torch.Tensor
    ) -> torch.Tensor:
        """Returns the negative log-ratio function evaluated each of 
        the current set of samples.
        
        Parameters
        ----------
        reference:
            The reference distribution.
        method:
            The method used to compute the ratio function. Can be
            'eratio' (exact) or 'aratio' (approximate).
        xs:
            An n * d matrix containing a set of samples from the 
            approximation domain.
        neglogliks:
            An n-dimensional vector containing the negative 
            log-likelihood evaluated at each sample.
        neglogpris:
            An n-dimensional vector containing the negative log-prior
            density evaluated at each sample.
        neglogfxs:
            An n-dimensional vector containing the negative logarithm
            of the density the samples are drawn from.

        Returns
        -------
        neglogratio:
            An n-dimensional vector containing the negative log-ratio 
            function evaluated for each sample.
            
        """
        return

    @abc.abstractmethod
    def ratio_func(
        func: Callable, 
        rs: torch.Tensor,
        irt_func: Callable,
        reference: Reference,
        method: str
    ) -> torch.Tensor:
        """Returns the negative log-ratio function associated with a 
        set of samples from the reference density.
        
        Parameters
        ----------
        func:
            User-defined function that returns the negative 
            log-likelihood and negative log-prior density of a sample 
            in the approximation domain.
        zs:
            The samples from the reference density.
        irt_func:
            Function that computes the inverse Rosenblatt transform.
        reference:
            The reference density.
        method:
            The method to use when computing the ratio function; can be
            `aratio` (approximate ratio) or `eratio` (exact ratio).
        
        Returns
        -------
        neglogratio: 
            An n-dimensional vector containing the negative log-ratio 
            function evaluated for each sample.

        """
        return
    
    # @abc.abstractmethod 
    # def set_init(self, mllkds, etol):
    #     return 
    
    @abc.abstractmethod
    def adapt_density(
        self,
        method: str, 
        neglogliks: torch.Tensor, 
        neglogpris: torch.Tensor, 
        neglogfxs: torch.Tensor
    ) -> None:
        """Determines the beta value associated with the next bridging 
        density.
        
        Parameters
        ----------
        method: 
            The method used to select the next bridging parameter. Can
            be `aratio` (approximate ratio) or `eratio` (exact ratio).
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
    
    @abc.abstractmethod
    def compute_log_weights(
        self,
        neglogliks: torch.Tensor,
        neglogpris: torch.Tensor, 
        neglogfxs: torch.Tensor
    ) -> torch.Tensor:
        """Returns the ratio of the next (k+1th) (possibly unnormalised) 
        target density to the current (kth) approximation for all 
        particles.

        TODO: finish docstring.
        """
        return

    @abc.abstractmethod
    def print_progress(
        self, 
        neglogliks: torch.Tensor, 
        neglogpris: torch.Tensor, 
        neglogfx: torch.Tensor
    ) -> None:
        return 
    
    # def eval(
    #     self, 
    #     func: Callable, 
    #     xs: torch.Tensor
    # ) -> tuple[float, float]:
    #     """Returns (quantities proportional to) the negative 
    #     log-likelihood and negative log-prior density associated with 
    #     a set of parameters. 
        
    #     Parameters
    #     ----------
    #     func: 
    #         Returns a quantity proportional to the log of the current 
    #         bridging density.
    #     xs: 
    #         The parameters at which to evaluate func.

    #     Returns
    #     -------
    #     :
    #         The negative log-likelihood and negative log-prior density
    #         associated with x.

    #     TODO: get rid of this.
            
    #     """
    #     return func(xs)