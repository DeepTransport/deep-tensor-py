from typing import List, Tuple

from torch import Tensor

from .bridge import Bridge
from ..debiasing.importance_sampling import estimate_ess_ratio
from ..preconditioners import Preconditioner
from ..target_functions import TargetFunc
from ..tools import compute_f_divergence


class Tempering(Bridge):
    r"""Likelihood tempering.
    
    The intermediate densities, $\{\pi_{k}(\theta)\}_{k=1}^{N}$, 
    generated using this approach take the form
    $$
        \pi_{k}(\theta) \propto (Q_{\sharp}\rho(\theta))^{1-\beta_{k}}\pi(\theta)^{\beta_{k}},
    $$
    where $Q_{\sharp}\rho(\cdot)$ denotes the pushforward of the 
    reference density, $\rho(\cdot)$, under the preconditioner, 
    $Q(\cdot)$, $\pi(\cdot)$ denotes the target density, and 
    $0 \leq \beta_{1} \leq \cdots \leq \beta_{N} = 1$.

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
        betas: List | Tensor | None = None, 
        ess_tol: float = 0.5, 
        ess_tol_init: float = 0.5,
        beta_factor: float = 1.05,
        min_beta: float = 1e-04,
        max_layers: int = 20
    ):
        
        if betas is not None:
            if abs(betas[-1] - 1.0) > 1e-6:
                msg = "Final beta value must be equal to 1."
                raise Exception(msg)
            if isinstance(betas, Tensor):
                betas = betas.tolist()
            self.betas = dict(enumerate(betas))
        else:
            self.betas = {}
        
        self.betas[-1] = 0.0
        self.ess_tol = ess_tol
        self.ess_tol_init = ess_tol_init
        self.beta_factor = beta_factor
        self.min_beta = min_beta
        self.init_beta = min_beta
        self.max_layers = max_layers
        self.is_adaptive = len(self.betas) == 1
        self.n_layers = 0
        self.initialised = False

        self._ratio_weight_funcs = {
            "aratio": self._compute_weights_aratio,
            "eratio": self._compute_weights_eratio
        }

        return
    
    @property 
    def is_last(self) -> bool:
        max_layers_reached = self.n_layers == self.max_layers
        final_beta_reached = abs(self.betas[self.n_layers-1] - 1.0) < 1e-6
        return bool(max_layers_reached or final_beta_reached)

    def initialise(
        self, 
        preconditioner: Preconditioner, 
        target_func: TargetFunc
    ) -> None:
        self.preconditioner = preconditioner
        self.reference = self.preconditioner.reference
        self.target_func = target_func
        self.initialised = True
        return
  
    def _compute_neglogbridges(
        self, 
        neglogref_us: Tensor,
        neglogfus: Tensor
    ) -> Tensor:
        
        k = self.n_layers
        neglogbridges = (
            + (1.0 - self.betas[k-1]) * neglogref_us 
            + self.betas[k-1] * neglogfus
        )
        return neglogbridges
    
    def _compute_weights_aratio(
        self,
        neglogref_us: Tensor, 
        neglogfus: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tensor:
        """Computes the ratio between the current bridging density and 
        the previous bridging density for each particle.
        """
        k = self.n_layers
        neglogweights = (
            + (self.betas[k-1] - self.betas[k]) * neglogref_us 
            + (self.betas[k] - self.betas[k-1]) * neglogfus
        )
        return neglogweights

    def _compute_weights_eratio(
        self,
        neglogref_us, 
        neglogfus, 
        neglogfus_dirt
    ) -> Tensor:
        """Computes the ratio between the current bridging density and 
        the DIRT approximation to the previous bridging density for 
        each particle.
        """
        k = self.n_layers
        neglogweights = (
            + (1.0 - self.betas[k]) * neglogref_us 
            + self.betas[k] * neglogfus
            - neglogfus_dirt
        )
        return neglogweights
    
    def _compute_ratio_func(
        self, 
        method: str,
        neglogref_rs: Tensor,
        neglogref_us: Tensor, 
        neglogfus: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tensor:

        neglogratios = self._ratio_weight_funcs[method](
            neglogref_us,
            neglogfus, 
            neglogfus_dirt
        ) + neglogref_rs
        return neglogratios
    
    def _compute_log_weights(
        self, 
        neglogrefs: Tensor,
        neglogfus: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        beta = self.betas[self.n_layers]
        log_weights = -beta*neglogfus - (1-beta)*neglogrefs + neglogfus_dirt
        return log_weights
    
    def ratio_func(
        self,
        method: str,
        rs: Tensor,
        us: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        
        if not self.initialised:
            raise Exception("Need to call self.initialise().")
        
        neglogref_rs = self.reference.eval_potential(rs)[0]
        neglogref_us = self.reference.eval_potential(us)[0]

        xs, neglogdets = self.apply_preconditioner(us)
        neglogfxs = self.target_func(xs)
        neglogfus = neglogfxs + neglogdets

        neglogratios = self._compute_ratio_func(
            method,
            neglogref_rs,
            neglogref_us,
            neglogfus,
            neglogfus_dirt
        )
        return neglogratios
    
    def _adapt_beta(
        self,
        neglogref_us: Tensor,
        neglogfus: Tensor,
        neglogfus_dirt: Tensor
    ):
        
        if self.n_layers == 0:
            self.betas[0] = self.init_beta
            return
        
        k = self.n_layers
        self.betas[k] = self.betas[k-1] * self.beta_factor

        while True:

            log_weights = self._compute_log_weights(
                neglogref_us, 
                neglogfus, 
                neglogfus_dirt
            )          
            if estimate_ess_ratio(log_weights) < self.ess_tol:
                self.betas[k] = min(self.betas[k], 1.0)
                break
            
            self.betas[k] *= self.beta_factor

        return
    
    def update(
        self, 
        us: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        neglogref_us = self.reference.eval_potential(us)[0]
        
        xs, neglogdets = self.apply_preconditioner(us)
        neglogfxs = self.target_func(xs)
        neglogfus = neglogfxs + neglogdets

        if self.is_adaptive:
            self._adapt_beta(neglogref_us, neglogfus, neglogfus_dirt)

        log_weights = self._compute_log_weights(
            neglogref_us, 
            neglogfus, 
            neglogfus_dirt
        )
        
        neglogbridges = self._compute_neglogbridges(
            neglogref_us, 
            neglogfus
        )
        
        return log_weights, neglogbridges

    def _get_diagnostics(
        self, 
        log_weights: Tensor,
        neglogfus: Tensor,
        neglogfus_dirt: Tensor
    ) -> List[str]:

        div_h2 = compute_f_divergence(-neglogfus_dirt, -neglogfus)
        ess = estimate_ess_ratio(log_weights)

        msg = [
            f"DHell: {div_h2.sqrt():.4f}",
            f"Beta: {self.betas[self.n_layers]:.4f}", 
            f"ESS: {ess:.4f}"
        ]
        return msg