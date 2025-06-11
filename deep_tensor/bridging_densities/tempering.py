from typing import Dict, List

import torch
from torch import Tensor

from .bridge import Bridge
from ..tools import estimate_ess_ratio, compute_f_divergence


class AbstractTempering(Bridge):

    @property 
    def betas(self) -> Tensor:
        return self._betas
    
    @betas.setter 
    def betas(self, value: Tensor) -> None:
        self._betas = value 
        return
    
    @property 
    def n_layers(self) -> int:
        return self._n_layers
    
    @n_layers.setter 
    def n_layers(self, value: int) -> None:
        self._n_layers = value 
        return
    
    @property 
    def max_layers(self) -> int:
        return self._max_layers
    
    @max_layers.setter 
    def max_layers(self, value: int) -> None:
        self._max_layers = value 
        return

    @property 
    def is_last(self) -> bool:
        max_layers_reached = self.n_layers == self.max_layers
        final_beta_reached = (self.betas[self.n_layers-1] - 1.0).abs() < 1e-6
        return max_layers_reached or final_beta_reached
    
    @property
    def params_dict(self) -> Dict:
        return {"betas": self.betas, "n_layers": self.n_layers}


class Tempering(AbstractTempering):
    r"""Likelihood tempering.
    
    The intermediate densities, $\{\pi_{k}(\theta)\}_{k=1}^{N}$, 
    generated using this approach take the form
    $$\pi_{k}(\theta) \propto (Q_{\sharp}\rho(\theta))^{1-\beta_{k}}\pi(\theta)^{\beta_{k}},$$
    where $Q_{\sharp}\rho(\cdot)$ denotes the pushforward of the 
    reference density, $\rho(\cdot)$, under the preconditioner, 
    $Q(\cdot)$, $\pi(\cdot)$ denotes the target density, and 
    $0 \leq \beta_{1} < \cdots < \beta_{N} = 1$.

    It is possible to provide this class with a set of $\beta$ values to 
    use. If these are not provided, they will be determined 
    automatically by finding the largest possible $\beta$, at each 
    iteration, such that the ESS of a reweighted set of samples 
    distributed according to (a TT approximation to) the previous 
    bridging density does not fall below a given value. 

    Parameters
    ----------
    betas:
        A set of $\beta$ values to use for the intermediate 
        distributions. If not specified, these will be determined 
        automatically.
    ess_tol:
        If selecting the $\beta$ values adaptively, the minimum 
        allowable ESS of the samples (distributed according to an 
        approximation of the previous bridging density) when selecting 
        the next bridging density. 
    ess_tol_init:
        If selecting the $\beta$ values adaptively, the minimum 
        allowable ESS of the samples when selecting the initial 
        bridging density.
    beta_factor:
        If selecting the $\beta$ values adaptively, the factor by which 
        to increase the current $\beta$ value by prior to checking 
        whether the ESS of the reweighted samples is sufficiently high.
    min_beta:
        If selecting the $\beta$ values adaptively, the minimum 
        allowable $\beta$ value.
    max_layers:
        If selecting the $\beta$ values adaptively, the maximum number 
        of layers to construct. Note that, if the maximum number of
        layers is reached, the final bridging density may not be the 
        target density.
        
    """

    def __init__(
        self, 
        betas: Tensor | None = None, 
        ess_tol: Tensor | float = 0.5, 
        ess_tol_init: Tensor | float = 0.5,
        beta_factor: Tensor | float = 1.05,
        min_beta: Tensor | float = 1e-4,
        max_layers: int = 20
    ):
        
        if betas is not None:
            if torch.abs(betas[-1] - 1.0) > 1e-6:
                msg = "Final beta value must be equal to 1."
                raise Exception(msg)
        else:
            betas = torch.tensor([])
        
        self.betas = betas
        self.ess_tol = torch.tensor(ess_tol)
        self.ess_tol_init = torch.tensor(ess_tol_init)
        self.beta_factor = torch.tensor(beta_factor)
        self.min_beta = torch.tensor(min_beta)
        self.init_beta = torch.tensor(min_beta)
        self.max_layers = max_layers
        self.is_adaptive = self.betas.numel() == 0
        self.n_layers = 0
        return
    
    def _set_init(self, neglogliks: Tensor) -> None:

        if not self.is_adaptive:
            return 

        beta = self.min_beta
        while True:
            log_ratios = -beta*self.beta_factor*neglogliks
            if estimate_ess_ratio(log_ratios) < self.ess_tol:
                beta = torch.minimum(torch.tensor(1.0), beta)
                self.init_beta = beta
                return
            beta *= self.beta_factor
    
    @staticmethod
    def _compute_ratio_weights(
        method,
        beta_p, 
        beta, 
        neglogrefs, 
        neglogfxs, 
        neglogfxs_dirt
    ) -> Tensor:
        
        if method == "aratio":
            log_weights = -(beta_p-beta)*neglogrefs - (beta-beta_p)*neglogfxs
        elif method == "eratio":
            log_weights = -(1-beta)*neglogrefs - beta*neglogfxs + neglogfxs_dirt
        return log_weights
    
    def _compute_log_weights(
        self, 
        neglogrefs: Tensor,
        neglogfxs: Tensor,
        neglogfxs_dirt: Tensor
    ) -> Tensor:
        beta = self.betas[self.n_layers]
        log_weights = -beta*neglogfxs - (1-beta)*neglogrefs + neglogfxs_dirt
        return log_weights

    def _adapt_density(
        self, 
        method: str, 
        neglogrefs: Tensor, 
        neglogfxs: Tensor, 
        neglogfxs_dirt: Tensor
    ) -> None:
        
        if not self.is_adaptive:
            return
            
        if self.n_layers == 0:
            self.betas = torch.tensor([self.init_beta])
            return
            
        beta_p = self.betas[self.n_layers-1]
        beta = beta_p * self.beta_factor

        while True:

            log_weights = Tempering._compute_ratio_weights(
                method, 
                beta_p, 
                beta * self.beta_factor, 
                neglogrefs, 
                neglogfxs, 
                neglogfxs_dirt
            )
            
            if estimate_ess_ratio(log_weights) < self.ess_tol:
                beta = torch.minimum(beta, torch.tensor(1.0))
                self.betas = torch.cat((self.betas, beta.reshape(1)))
                return
            
            beta *= self.beta_factor

    def _get_ratio_func(
        self, 
        method: str,
        neglogrefs_rs: Tensor,
        neglogrefs: Tensor, 
        neglogfxs: Tensor, 
        neglogfxs_dirt: Tensor
    ) -> Tensor:
        
        beta = self.betas[self.n_layers]

        if self.n_layers == 0:
            neglogratios = beta*neglogfxs + (1-beta)*neglogrefs
            return neglogratios

        beta_p = self.betas[self.n_layers-1]

        log_weights = Tempering._compute_ratio_weights(
            method, 
            beta_p, 
            beta, 
            neglogrefs, 
            neglogfxs, 
            neglogfxs_dirt
        )
        neglogratios = -log_weights + neglogrefs_rs
        return neglogratios

    def _get_diagnostics(
        self, 
        log_weights: Tensor,
        neglogrefs: Tensor,
        neglogfxs: Tensor,
        neglogfxs_dirt: Tensor
    ) -> List[str]:

        ess = estimate_ess_ratio(log_weights)

        msg = [
            f"Beta: {self.betas[self.n_layers]:.4f}", 
            f"ESS: {ess:.4f}"
        ]

        if self.n_layers > 0:
            beta_p = self.betas[self.n_layers-1]
            log_approx = -neglogfxs_dirt
            log_target = -beta_p*neglogfxs - (1-beta_p)*neglogrefs
            div_h2 = compute_f_divergence(log_approx, log_target)
            msg.append(f"DHell: {div_h2.sqrt():.4f}")

        return msg
    

class SavedTempering(AbstractTempering):

    def __init__(self, betas: Tensor, n_layers: int):
        self.betas = betas 
        self.n_layers = n_layers
        return
    
    def _compute_log_weights(self, neglogliks, neglogpris, neglogfxs):
        raise NotImplementedError()
    
    def _get_ratio_func(self, reference, method, rs, neglogliks, neglogpris, neglogfxs):
        raise NotImplementedError()