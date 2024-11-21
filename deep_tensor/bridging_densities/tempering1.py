from typing import Callable

import torch

from .single_beta import SingleBeta
from ..references import Reference
from ..tools import compute_ess_ratio
from ..utils import info


class Tempering1(SingleBeta):

    def __init__(
        self, 
        betas: torch.Tensor=torch.tensor([]), 
        **kwargs
    ):

        super().__init__(betas, **kwargs)
        self.betas = torch.sort(betas)[0]
        self._is_adaptive = self.betas.numel() == 0
        self._num_layers = 0
        return
    
    @property
    def is_adaptive(self) -> bool:
        return self._is_adaptive

    @property 
    def num_layers(self) -> int:
        return self._num_layers
    
    @num_layers.setter
    def num_layers(self, value):
        self._num_layers = value
    
    def adapt_density(
        self, 
        method: str, 
        neglogliks: torch.Tensor, 
        neglogpris: torch.Tensor, 
        neglogfs: torch.Tensor
    ) -> None:
        """Determines the beta value associated with the next bridging 
        density.
        
        Parameters
        ----------
        method: 
            The method used to select the next bridging parameter. Can
            be `aratio` (approximate ratio) or `eratio` (exact ratio).
        neglogliks: 
            The negative log-likelihood of each sample.
        neglogpris:
            The negative log-prior density of each sample.
        neglogfs:
            The negative log-density of the current approximation to 
            the target distribution for each sample.
        
        """
        
        if not self.is_adaptive:
            return
            
        if self.num_layers == 0:
            beta = self.init_beta 
            self.betas = torch.cat((self.betas, beta.reshape(1)))
            return
            
        beta_prev = self.betas[self.num_layers-1]
        beta = max(beta_prev, self.min_beta)

        # TODO: tidy this up (the while loop is pretty much the same in both cases.)
        if method == "aratio":
            ess = compute_ess_ratio(-(beta-beta_prev)*neglogliks)
            while ess > self.ess_tol:
                beta *= self.beta_factor
                ess = compute_ess_ratio(-(beta-beta_prev)*neglogliks)
        
        elif method == "eratio":
            # Compute ess over sample size
            ess = compute_ess_ratio(-beta*neglogliks-neglogpris+neglogfs)
            while ess > self.ess_tol:
                beta *= self.beta_factor
                ess = compute_ess_ratio(-beta*neglogliks-neglogpris+neglogfs)

        beta = min(1.0, beta)
        
        self.betas = torch.cat((self.betas, beta.reshape(1)))
        return

    def get_ratio_func(
        self, 
        reference: Reference, 
        method: str,
        zs: torch.Tensor,
        neglogliks: torch.Tensor, 
        neglogpris: torch.Tensor, 
        neglogfs: torch.Tensor # the current approximate density 
        # evaluated at each of the current samples (pushed forward 
        # under the current mapping)
    ) -> torch.Tensor:
        """Returns the negative log-ratio function."""
        
        beta = self.betas[self.num_layers]

        if self.num_layers == 0:
            ratio = beta * neglogliks + neglogpris
            return ratio
        
        # Compute the reference density at each z value
        logfz = reference.log_joint_pdf(zs)
        beta_prev = self.betas[self.num_layers-1]

        if method == "aratio":
            ratio = (beta-beta_prev) * neglogliks - logfz
        elif method == "eratio":
            ratio = beta * neglogliks + neglogpris - neglogfs - logfz
        
        return ratio
    
    def ratio_func(
        self, 
        func: Callable, 
        zs: torch.Tensor,
        irt_func: Callable,
        reference: Reference,
        method: str
    ) -> torch.Tensor:
        """Returns the negative log-ratio function associated with a 
        set of samples from the reference domain.
        
        Parameters
        ----------
        func:
            User-defined function that returns the negative 
            log-likelihood and negative log-prior density of a sample 
            in the approximation domain.
        zs:
            The samples from the reference domain.
        irt_func:
            Function that computes the inverse Rosenblatt transform.
        reference:
            The reference distribution.
        method:
            The method to use when computing the ratio function; can be
            `aratio` (approximate ratio) or `eratio` (exact ratio).
        
        Returns
        -------
        : 
            The negative log-ratio function evaluated for each sample.

        """
        
        # Push samples forward to the approximation domain
        xs, neglogfs = irt_func(zs)
        neglogliks, neglogpris = func(xs)
        
        f = self.get_ratio_func(
            reference, 
            method, 
            zs, 
            neglogliks, 
            neglogpris, 
            neglogfs
        )

        return f
    
    def compute_log_weights(
        self, 
        neglogliks: torch.Tensor,
        neglogpris: torch.Tensor,
        neglogfs: torch.Tensor
    ) -> torch.Tensor:
        """Returns the logarithm of the current density function being
        approximated at each of a set of samples.
        """
        
        log_weights = (- self.betas[self.num_layers]*neglogliks
                       - neglogpris 
                       + neglogfs)
        
        return log_weights

    def print_progress(
        self, 
        log_weights: torch.Tensor
    ) -> None:

        ess = compute_ess_ratio(log_weights)

        if self.num_layers == 0:
            msg = [
                f"Iter: {self.num_layers}", 
                f"beta: {self.betas[self.num_layers-1]:.4f}", 
                f"ESS: {ess:.4f}"
            ]
            info(" | ".join(msg))
            return
        
        raise NotImplementedError()

        return