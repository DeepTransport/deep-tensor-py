from typing import Callable

import torch
from torch import Tensor

from .single_beta import SingleBeta
from ..references import Reference
from ..tools import compute_ess_ratio, compute_f_divergence
from ..tools.printing import dirt_info


class Tempering1(SingleBeta):
    r"""Likelihood tempering.
    
    The intermediate densities, $\{\pi_{k}(\theta)\}_{k=1}^{N}$, 
    generated using this approach take the form
    $$\pi_{k}(\theta) \propto \pi_{0}(\theta)\mathcal{L}(\theta; y)^{\beta_{k}},$$
    where $\pi_{0}(\,\cdot\,)$ denotes the prior, $\mathcal{L}(\,\cdot\,; y)$ 
    denotes the likelihood, and $0 \leq \beta_{1} < \cdots < \beta_{N} = 1$.

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
        
    """

    def __init__(
        self, 
        betas: Tensor|None = None, 
        ess_tol: Tensor|float = 0.5, 
        ess_tol_init: Tensor|float = 0.5,
        beta_factor: Tensor|float = 1.05,
        min_beta: Tensor|float = 1e-4
    ):
        
        if betas == None:
            betas = torch.tensor([])

        SingleBeta.__init__(
            self, 
            betas, 
            ess_tol, 
            ess_tol_init, 
            beta_factor, 
            min_beta
        )

        self.is_adaptive = self.betas.numel() == 0
        self.n_layers = 0
        return

    @property 
    def is_last(self) -> bool:
        return (self.betas[self.n_layers-1] - 1.0).abs() < 1e-6
    
    @property
    def is_adaptive(self) -> bool:
        return self._is_adaptive
    
    @is_adaptive.setter 
    def is_adaptive(self, value: bool) -> None:
        self._is_adaptive = value 
        return

    @property 
    def n_layers(self) -> int:
        return self._n_layers
    
    @n_layers.setter
    def n_layers(self, value: int) -> None:
        self._n_layers = value
        return
    
    def set_init(self, neglogliks: Tensor) -> None:

        if not self.is_adaptive:
            return 

        beta = self.min_beta
        while True:
            log_ratios = -beta*self.beta_factor*neglogliks
            if compute_ess_ratio(log_ratios) < self.ess_tol:
                beta = torch.minimum(torch.tensor(1.0), beta)
                self.init_beta = beta
                return
            beta *= self.beta_factor
    
    @staticmethod
    def compute_weights(
        method,
        beta_p, 
        beta, 
        neglogliks, 
        neglogpris, 
        neglogfxs
    ) -> Tensor:
        
        if method == "aratio":
            return -(beta-beta_p) * neglogliks
        elif method == "eratio":
            return -beta*neglogliks - neglogpris + neglogfxs
        raise Exception("Unknown ratio method.")

    def adapt_density(
        self, 
        method: str, 
        neglogliks: Tensor, 
        neglogpris: Tensor, 
        neglogfxs: Tensor
    ) -> None:
        
        if not self.is_adaptive:
            return
            
        if self.n_layers == 0:
            self.betas = torch.tensor([self.init_beta])
            return
            
        beta_p = self.betas[self.n_layers-1]
        beta = beta_p * self.beta_factor

        while True:

            log_weights = Tempering1.compute_weights(
                method, 
                beta_p, 
                beta * self.beta_factor, 
                neglogliks, 
                neglogpris, 
                neglogfxs
            )
            
            if compute_ess_ratio(log_weights) < self.ess_tol:
                beta = torch.minimum(beta, torch.tensor(1.0))
                self.betas = torch.cat((self.betas, beta.reshape(1)))
                return
            
            beta *= self.beta_factor

    def get_ratio_func(
        self, 
        reference: Reference, 
        method: str,
        rs: Tensor,
        neglogliks: Tensor, 
        neglogpris: Tensor, 
        neglogfxs: Tensor
    ) -> Tensor:
        
        beta = self.betas[self.n_layers]

        if self.n_layers == 0:
            neglogratios = beta*neglogliks + neglogpris
            return neglogratios

        beta_p = self.betas[self.n_layers-1]

        log_weights = Tempering1.compute_weights(
            method, 
            beta_p, 
            beta, 
            neglogliks, 
            neglogpris, 
            neglogfxs
        )
        neglogrefs = reference.eval_potential(rs)[0]
        neglogratios = -log_weights + neglogrefs
        return neglogratios
    
    # def ratio_func(
    #     self, 
    #     func: Callable, 
    #     rs: Tensor,
    #     irt_func: Callable,
    #     reference: Reference,
    #     method: str
    # ) -> Tensor:
        
    #     # Push samples forward to the approximation of the current target
    #     xs, neglogfxs = irt_func(rs)
    #     neglogliks, neglogpris = func(xs)
    #     neglogratios = self.get_ratio_func(
    #         reference, 
    #         method, 
    #         rs, 
    #         neglogliks, 
    #         neglogpris, 
    #         neglogfxs
    #     )
    #     return neglogratios
    
    def compute_log_weights(
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
        beta = self.betas[self.n_layers]
        log_weights = -beta*neglogliks - neglogpris + neglogfxs
        return log_weights

    def print_progress(
        self, 
        log_weights: Tensor,
        neglogliks: Tensor,
        neglogpris: Tensor,
        neglogfxs: Tensor
    ) -> None:
        """Prints some information about the current bridging density.
        """

        ess = compute_ess_ratio(log_weights)

        msg = [
            f"Iter: {self.n_layers}", 
            f"beta: {self.betas[self.n_layers]:.4f}", 
            f"ESS: {ess:.4f}"
        ]

        if self.n_layers > 0:
            beta_p = self.betas[self.n_layers-1]
            log_approx = -neglogfxs
            log_target = -beta_p*neglogliks - neglogpris
            div_h2 = compute_f_divergence(log_approx, log_target)
            msg.append(f"DHell: {div_h2.sqrt():.4f}")

        dirt_info(" | ".join(msg))
        return