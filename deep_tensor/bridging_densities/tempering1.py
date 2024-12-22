from typing import Callable

import torch

from .single_beta import SingleBeta
from ..references import Reference
from ..tools import compute_ess_ratio, compute_f_divergence
from ..utils import dirt_info


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
        
        beta = self.betas[self.num_layers]

        if self.num_layers == 0:
            neglogratio = beta*neglogliks + neglogpris  # neglogfrs == neglogpris in the first instance, also beta_prev=0
            return neglogratio
        
        # Compute the reference density at each value of xs
        neglogfrs = -reference.log_joint_pdf(xs)[0]
        beta_prev = self.betas[self.num_layers-1]

        if method == "eratio":
            # beta*neglogliks + neglogpris = exact evaluations of next target of samples under current composition of mappings
            # neglogfrs = density of reference evaluated at each sample
            # neglogfxs = current approximation density (that the samples are drawn from) evaluated at each sample
            # TODO: figure out where omega went here. I assume it's somehow lumped in to neglogfxs..?
            neglogratios = (beta*neglogliks + neglogpris + neglogfrs
                            - neglogfxs)
        elif method == "aratio":
            neglogratios = (beta-beta_prev)*neglogliks + neglogfrs
        
        return neglogratios
    
    def ratio_func(
        self, 
        func: Callable, 
        rs: torch.Tensor,
        irt_func: Callable,
        reference: Reference,
        method: str
    ) -> torch.Tensor:
        
        # Push samples forward to the approximation of the current target
        xs, neglogfxs = irt_func(rs)
        neglogliks, neglogpris = func(xs)
        
        neglogratios = self.get_ratio_func(
            reference, 
            method, 
            rs, 
            neglogliks, 
            neglogpris, 
            neglogfxs
        )

        return neglogratios
    
    def compute_log_weights(
        self, 
        neglogliks: torch.Tensor,
        neglogpris: torch.Tensor,
        neglogfxs: torch.Tensor
    ) -> torch.Tensor:

        log_weights = (-self.betas[self.num_layers]*neglogliks - neglogpris
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
        
            beta = self.betas[self.num_layers-1]
            log_proposal = -neglogfxs
            log_target = -beta*neglogliks - neglogpris

            # Estimate square Hellinger distance between current (kth)
            # approximation and previous (kth) target density
            div_h2 = compute_f_divergence(log_proposal, log_target)[1]

            # f"prev beta: {self.betas[self.num_layers-1]:.4f}"
            msg.append(f"DHell: {div_h2.sqrt()[0]:.4f}")

        dirt_info(" | ".join(msg))
        return