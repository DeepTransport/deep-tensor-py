from types import NoneType
from typing import List, Tuple

from torch import Tensor

from .bridge import Bridge
from ..debiasing.importance_sampling import estimate_ess_ratio
from ..preconditioners import Preconditioner
from ..target_functions import TargetFunc
from ..tools import estimate_dhell


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
    beta_factor:
        If selecting the $\beta$ values adaptively, the factor by which 
        to increase the current $\beta$ value prior to checking 
        whether the ESS of the reweighted samples is sufficiently high.
    init_beta:
        If selecting the $\beta$ values adaptively, the initial $\beta$ 
        value to use.
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
        beta_factor: float = 1.05,
        init_beta: float = 1e-04,
        max_layers: int = 20
    ):
        
        if betas is not None:
            if abs(betas[-1] - 1.0) > 1e-6:
                msg = "Final beta value must be equal to 1."
                raise Exception(msg)
            if isinstance(betas, Tensor):
                betas = betas.tolist()
            self._betas = dict(enumerate(betas))
        else:
            self._betas = {}
        
        self._betas[-1] = 0.0
        self._ess_tol = ess_tol
        self._beta_factor = beta_factor
        self._init_beta = init_beta
        self._max_layers = max_layers
        self._is_adaptive = len(self._betas) == 1
        self._num_layers = 0
        self._initialised = False

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
    def _is_last(self) -> bool:
        max_layers_reached = self._num_layers == self._max_layers
        final_beta_reached = abs(self._betas[self._num_layers-1] - 1.0) < 1e-6
        return bool(max_layers_reached or final_beta_reached)
    
    def _reset(self) -> None:
        self._num_layers = 0
        self._initialised = False
        if self._is_adaptive:
            self._betas = {-1: 0.0}
        return

    def _initialise(
        self, 
        preconditioner: Preconditioner, 
        target_func: TargetFunc
    ) -> None:
        Bridge._initialise(self, preconditioner, target_func)
        self._initialised = True
        return
    
    def _eval_neglogweights_aratio(
        self,
        neglogref_us: Tensor, 
        neglogfus: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tensor:
        """Computes the negative logarithm of the ratio between the 
        current bridging density and the previous bridging density for 
        each particle.
        """
        k = self._num_layers
        neglogweights = (
            + (self._betas[k-1] - self._betas[k]) * neglogref_us 
            + (self._betas[k] - self._betas[k-1]) * neglogfus
        )
        return neglogweights
    
    def _grad_neglogweights_aratio(
        self,
        neglogref_us: Tensor,
        grad_neglogref_us: Tensor,
        neglogfus: Tensor, 
        grad_neglogfus: Tensor,
        neglogfus_dirt: Tensor,
        grad_neglogfus_dirt: Tensor | None
    ) -> Tuple[Tensor, Tensor]:
        k = self._num_layers
        neglogweights = self._eval_neglogweights_aratio(
            neglogref_us, 
            neglogfus, 
            neglogfus_dirt
        )
        grad_neglogweights = (
            + (self._betas[k-1] - self._betas[k]) * grad_neglogref_us 
            + (self._betas[k] - self._betas[k-1]) * grad_neglogfus
        )
        return neglogweights, grad_neglogweights

    def _eval_neglogweights_eratio(
        self,
        neglogref_us: Tensor, 
        neglogfus: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tensor:
        k = self._num_layers
        neglogweights = (
            + (1.0 - self._betas[k]) * neglogref_us 
            + self._betas[k] * neglogfus
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
        k = self._num_layers
        neglogweights = self._eval_neglogweights_eratio(
            neglogref_us,
            neglogfus,
            neglogfus_dirt
        )
        grad_neglogweights = (
            + (1.0 - self._betas[k]) * grad_neglogref_us 
            + self._betas[k] * grad_neglogfus
            - grad_neglogfus_dirt
        )
        return neglogweights, grad_neglogweights
    
    def _compute_log_weights(
        self, 
        neglogrefs: Tensor,
        neglogfus: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        beta = self._betas[self._num_layers]
        log_weights = -beta*neglogfus - (1-beta)*neglogrefs + neglogfus_dirt
        return log_weights
    
    def _eval_neglogratio(
        self,
        method: str,
        rs: Tensor,
        us: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        
        if not self._initialised:
            raise Exception("Need to call self.initialise().")
        
        neglogref_rs = self._reference.eval_potential(rs)[0]
        neglogref_us = self._reference.eval_potential(us)[0]
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
        grad_neglogfus_dirt: Tensor | None,
        dudrs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        if grad_neglogfus_dirt is None and method == "eratio":
            msg = (
                "If method==`eratio`, the gradient of the DIRT density " 
                "must be passed in."
            )
            raise Exception(msg)
        
        # TODO: finite difference check on the output!!
        
        neglogref_rs, grad_neglogref_rs = self._reference._eval_potential_unnormalised(rs)
        neglogref_us, grad_neglogref_us = self._reference._eval_potential_unnormalised(us)

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
        k = num_layers if num_layers is not None else self._num_layers
        beta = self._betas[k]
        neglogbridges = (1.0 - beta) * neglogref_us + beta * neglogfus
        return neglogbridges
    
    def _grad_neglogbridge(
        self, 
        us: Tensor,
        dudrs: Tensor
    ) -> Tuple[Tensor, Tensor]:

        beta = self._betas[self._num_layers]

        neglogref_us, grad_neglogref_us = self._reference._eval_potential_unnormalised(us)
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
        
        if self._num_layers == 0:
            self._betas[0] = self._init_beta
            return
        
        k = self._num_layers
        self._betas[k] = self._betas[k-1] * self._beta_factor

        while True:

            log_weights = self._compute_log_weights(
                neglogref_us, 
                neglogfus, 
                neglogfus_dirt
            )          
            if estimate_ess_ratio(log_weights) < self._ess_tol:
                self._betas[k] = min(self._betas[k], 1.0)
                break
            
            self._betas[k] *= self._beta_factor

        return
    
    def _update(
        self, 
        us: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        neglogref_us = self._reference.eval_potential(us)[0]
        neglogfus = self._eval_pullback(us)

        if self._is_adaptive:
            self._adapt_beta(neglogref_us, neglogfus, neglogfus_dirt)

        log_weights = self._compute_log_weights(
            neglogref_us, 
            neglogfus, 
            neglogfus_dirt
        )

        neglogbridges = self._eval_neglogbridge(
            neglogref_us, 
            neglogfus,
            num_layers=self._num_layers-1
        )
        
        return log_weights, neglogbridges
    
    def _get_diagnostics(
        self, 
        log_weights: Tensor | None,
        neglogfus: Tensor | None,
        neglogfus_dirt: Tensor | None
    ) -> List[str]:
        
        msg = [f"Beta: {self._betas[self._num_layers]:.4f}"]

        if (isinstance(log_weights, NoneType) 
            or isinstance(neglogfus, NoneType)
            or isinstance(neglogfus_dirt, NoneType)): 
            return msg

        dhell = estimate_dhell(neglogfus_dirt, neglogfus)
        ess = estimate_ess_ratio(log_weights)
        msg += [f"DHell: {dhell:.4f}", f"ESS: {ess:.4f}"]
        return msg