from typing import Callable

import torch

from .single_beta import SingleBeta
from ..references import Reference
from ..tools import compute_ess_ratio, compute_f_divergence
from ..utils import info


class Tempering1(SingleBeta):

    def __init__(
        self, 
        betas: torch.Tensor|None=None, 
        **kwargs
    ):
        
        if betas == None:
            betas = torch.tensor([])

        super().__init__(betas, **kwargs)
        self.betas = torch.sort(betas)[0]
        self._is_adaptive = self.betas.numel() == 0
        self._num_layers = 0
        return
    
    @property
    def is_adaptive(self) -> bool:
        return self._is_adaptive

    @property 
    def is_last(self) -> bool:
        return torch.abs(self.betas[self.num_layers] - 1.0) < 1e-6

    @property 
    def num_layers(self) -> int:
        return self._num_layers
    
    @num_layers.setter
    def num_layers(self, value):
        self._num_layers = value
        return
    
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
            An n-dimensional vector containging the negative 
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
        
        if not self.is_adaptive:
            return
            
        if self.num_layers == 0:
            beta = self.init_beta 
            self.betas = torch.cat((self.betas, beta.reshape(1)))
            return
            
        beta_prev = self.betas[self.num_layers-1]
        beta = torch.maximum(beta_prev, self.min_beta).clone()

        if method == "aratio":
            log_weights = -(beta-beta_prev)*neglogliks
            while compute_ess_ratio(log_weights) > self.ess_tol:
                beta *= self.beta_factor
                log_weights = -(beta-beta_prev)*neglogliks
        
        elif method == "eratio":
            log_weights = -beta*neglogliks - neglogpris + neglogfxs
            while compute_ess_ratio(log_weights) > self.ess_tol:
                beta *= self.beta_factor
                log_weights = -beta*neglogliks - neglogpris + neglogfxs

        beta = torch.minimum(beta, torch.tensor(1.0))
        self.betas = torch.cat((self.betas, beta.reshape(1)))
        return

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
            The negative logarithm of the ratio function evaluated for
            each sample.
            
        """
        
        beta = self.betas[self.num_layers]

        if self.num_layers == 0:
            neglogratio = beta*neglogliks + neglogpris
            return neglogratio
        
        # Compute the reference density at each value of xs
        logfrs = reference.log_joint_pdf(xs)[0]
        beta_prev = self.betas[self.num_layers-1]

        if method == "eratio":  # TODO: match these to the paper
            neglogratio = beta*neglogliks + neglogpris - neglogfxs - logfrs
        elif method == "aratio":
            neglogratio = (beta-beta_prev)*neglogliks - logfrs
        
        return neglogratio
    
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
        neglogfxs: torch.Tensor
    ) -> torch.Tensor:
        """Returns the logarithm of the current density function being
        approximated at each of a set of samples.
        """
        
        log_weights = (- self.betas[self.num_layers] * neglogliks
                       - neglogpris 
                       + neglogfxs)
        
        return log_weights

    def print_progress(
        self, 
        log_weights: torch.Tensor,
        neglogliks: torch.Tensor,
        neglogpris: torch.Tensor,
        neglogfxs: torch.Tensor
    ) -> None:

        ess = compute_ess_ratio(log_weights)

        msg = [
            f"Iter: {self.num_layers}", 
            f"beta: {self.betas[self.num_layers]:.4f}", 
            f"ESS: {ess:.4f}"
        ]

        if self.num_layers > 0:
        
            lp_ref = -neglogfxs
            lp = -self.betas[self.num_layers-1] * neglogliks - neglogpris
            div_h2 = compute_f_divergence(lp_ref, lp)[1]

            msg = msg[:1] + [
                f"DHell: {div_h2.sqrt()[0]:.4f}",
                f"prev beta: {self.betas[self.num_layers-1]:.4f}"
            ] + msg[1:]

        info(" | ".join(msg))
        return