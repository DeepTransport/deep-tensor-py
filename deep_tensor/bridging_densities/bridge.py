import abc
from typing import Callable

import torch


class Bridge(abc.ABC):

    @property 
    @abc.abstractmethod 
    def is_adaptive(self) -> bool:
        return

    @property
    @abc.abstractmethod
    def num_layers(self) -> int:
        return

    @abc.abstractmethod
    def get_ratio_func(self, dirt, z, mllkds, mlps, mlogfx):
        return

    @abc.abstractmethod
    def ratio_func(self, func: Callable, dirt, z) -> torch.Tensor:
        """Evaluates the potential function of the current bridging
        density. Wrapper for `get_ratio_func`. TODO: need to re-write 
        this / change the input arguments.

        Parameters
        ----------
        func: 
            Returns the negative log-likelihood and negative log-prior 
            density associated with a sample.
        beta: TODO: fix this
            The current bridging parameter.
        z:
            Reference random variable.
        
        Returns
        -------
        :
            Ratio function.
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
        neglogfs: torch.Tensor
    ) -> None:
        return
    
    @abc.abstractmethod
    def print_progress(
        self, 
        neglogliks: torch.Tensor, 
        neglogpris: torch.Tensor, 
        neglogfx: torch.Tensor
    ):
        return 
    
    def eval(self, func: Callable, x: torch.Tensor) -> tuple[float, float]:
        """Returns (quantities proportional to) the negative 
        log-likelihood and negative log-prior density associated with 
        a set of parameters. 
        
        Parameters
        ----------
        func: 
            Returns a quantity proportional to the log of the current 
            bridging density.
        x: 
            The parameters at which to evaluate func.

        Returns
        -------
        :
            The negative log-likelihood and negative log-prior density
            associated with x.
            
        """
        return func(x)

    # @abc.abstractmethod
    # def islast(self):
    #     return