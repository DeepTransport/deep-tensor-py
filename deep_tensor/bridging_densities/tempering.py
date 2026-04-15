from types import NoneType
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
        self.num_layers = 0
        self.initialised = False

        self._ratio_weight_funcs = {
            "aratio": self._eval_neglogweights_aratio,
            "eratio": self._eval_neglogweights_eratio
        }

        self._grad_neglogweight_funcs = {
            "aratio": self._grad_neglogweights_aratio,
            "eratio": self._grad_neglogweights_eratio
        }

        return
    
    @property 
    def is_last(self) -> bool:
        max_layers_reached = self.num_layers == self.max_layers
        final_beta_reached = abs(self.betas[self.num_layers-1] - 1.0) < 1e-6
        return bool(max_layers_reached or final_beta_reached)
    
    def reset(self) -> None:
        self.num_layers = 0
        self.initialised = False
        if self.is_adaptive:
            self.betas = {-1: 0.0}
        return

    def initialise(
        self, 
        preconditioner: Preconditioner, 
        target_func: TargetFunc
    ) -> None:
        Bridge.initialise(self, preconditioner, target_func)
        self.initialised = True
        return
    
    def _eval_neglogweights_aratio(
        self,
        neglogref_us: Tensor, 
        neglogfus: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tensor:
        """Computes the ratio between the current bridging density and 
        the previous bridging density for each particle.
        """
        k = self.num_layers
        neglogweights = (
            + (self.betas[k-1] - self.betas[k]) * neglogref_us 
            + (self.betas[k] - self.betas[k-1]) * neglogfus
        )
        return neglogweights
    
    def _grad_neglogweights_aratio(
        self,
        neglogref_us: Tensor,
        grad_neglogref_us: Tensor,
        neglogfus: Tensor, 
        grad_neglogfus: Tensor,
        neglogfus_dirt: Tensor,
        grad_neglogfus_dirt: Tensor
    ) -> Tuple[Tensor, Tensor]:
        k = self.num_layers
        neglogweights = self._eval_neglogweights_aratio(
            neglogref_us, 
            neglogfus, 
            neglogfus_dirt
        )
        grad_neglogweights = (
            + (self.betas[k-1] - self.betas[k]) * grad_neglogref_us 
            + (self.betas[k] - self.betas[k-1]) * grad_neglogfus
        )
        return neglogweights, grad_neglogweights

    def _eval_neglogweights_eratio(
        self,
        neglogref_us: Tensor, 
        neglogfus: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tensor:
        k = self.num_layers
        neglogweights = (
            + (1.0 - self.betas[k]) * neglogref_us 
            + self.betas[k] * neglogfus
            - neglogfus_dirt
        )
        return neglogweights
    
    def _grad_neglogweights_eratio(
        self,
        neglogref_us: Tensor,
        grad_neglogref_us: Tensor,
        neglogfus: Tensor, 
        grad_neglogfus: Tensor,
        neglogfus_dirt: Tensor,
        grad_neglogfus_dirt: Tensor
    ) -> Tuple[Tensor, Tensor]:
        k = self.num_layers
        neglogweights = self._eval_neglogweights_eratio(
            neglogref_us,
            neglogfus,
            neglogfus_dirt
        )
        grad_neglogweights = (
            + (1.0 - self.betas[k]) * grad_neglogref_us 
            + self.betas[k] * grad_neglogfus
            - grad_neglogfus_dirt
        )
        return neglogweights, grad_neglogweights
    
    def _compute_log_weights(
        self, 
        neglogrefs: Tensor,
        neglogfus: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        beta = self.betas[self.num_layers]
        log_weights = -beta*neglogfus - (1-beta)*neglogrefs + neglogfus_dirt
        return log_weights
    
    def _eval_neglogratio(
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
        neglogfus = self._eval_pullback(us)

        neglogratios = self._ratio_weight_funcs[method](
            neglogref_us,
            neglogfus, 
            neglogfus_dirt
        ) + neglogref_rs
        return neglogratios
    
    def _grad_neglogratio(
        self,
        method: str,
        rs: Tensor,
        us: Tensor,
        neglogfus_dirt: Tensor,
        grad_neglogfus_dirt: Tensor,
        dudrs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        # TODO: finite difference check on the output!!
        
        neglogref_rs, grad_neglogref_rs = self.reference.eval_potential_unnormalised(rs)
        neglogref_us, grad_neglogref_us = self.reference.eval_potential_unnormalised(us)

        neglogfus, grad_neglogfus = self._grad_pullback(us)

        neglogweights, grad_neglogweights = self._grad_neglogweight_funcs[method](
            neglogref_us,
            grad_neglogref_us,
            neglogfus,
            grad_neglogfus,
            neglogfus_dirt,
            grad_neglogfus_dirt
        )
        grad_neglogweights = self._grad_chain(grad_neglogweights, dudrs)
        
        neglogratios = neglogweights + neglogref_rs 
        grad_neglogratios = grad_neglogweights + grad_neglogref_rs
        return neglogratios, grad_neglogratios

    def _eval_neglogbridge(
        self, 
        neglogref_us: Tensor,
        neglogfus: Tensor,
        num_layers: int | None = None  # in case we want to evaluate a previous density
    ) -> Tensor:
        k = num_layers if num_layers is not None else self.num_layers
        beta = self.betas[k]
        neglogbridges = (1.0 - beta) * neglogref_us + beta * neglogfus
        return neglogbridges
    
    def _grad_neglogbridge(
        self, 
        us: Tensor,
        dudrs: Tensor
    ) -> Tuple[Tensor, Tensor]:

        beta = self.betas[self.num_layers]

        neglogref_us, grad_neglogref_us = self.reference.eval_potential_unnormalised(us)
        neglogfus, grad_neglogfus = self._grad_pullback(us)

        neglogbridges = (1.0 - beta) * neglogref_us + beta * neglogfus

        # Compute gradient w.r.t. u
        grad_neglogbridges = (
            (1.0 - beta) * grad_neglogref_us 
            + beta * grad_neglogfus
        )

        # Change variable such that gradient is w.r.t. r 
        grad_neglogbridges = self._grad_chain(grad_neglogbridges, dudrs)
        
        return neglogbridges, grad_neglogbridges
    
    def _adapt_beta(
        self,
        neglogref_us: Tensor,
        neglogfus: Tensor,
        neglogfus_dirt: Tensor
    ):
        
        if self.num_layers == 0:
            self.betas[0] = self.init_beta
            return
        
        k = self.num_layers
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
        neglogfus = self._eval_pullback(us)

        if self.is_adaptive:
            self._adapt_beta(neglogref_us, neglogfus, neglogfus_dirt)

        log_weights = self._compute_log_weights(
            neglogref_us, 
            neglogfus, 
            neglogfus_dirt
        )

        neglogbridges = self._eval_neglogbridge(
            neglogref_us, 
            neglogfus,
            num_layers=self.num_layers-1
        )
        
        return log_weights, neglogbridges
    
    def _get_diagnostics(
        self, 
        log_weights: Tensor | None,
        neglogfus: Tensor | None,
        neglogfus_dirt: Tensor | None
    ) -> List[str]:
        
        msg = [f"Beta: {self.betas[self.num_layers]:.4f}"]

        if (isinstance(log_weights, NoneType) 
            or isinstance(neglogfus, NoneType)
            or isinstance(neglogfus_dirt, NoneType)): 
            return msg

        div_h2 = compute_f_divergence(-neglogfus_dirt, -neglogfus)
        ess = estimate_ess_ratio(log_weights)
        msg += [f"DHell: {div_h2.sqrt():.4f}", f"ESS: {ess:.4f}"]
        return msg