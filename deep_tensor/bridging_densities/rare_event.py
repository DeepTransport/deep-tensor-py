import abc
import math
from types import NoneType
from typing import Dict, List, Sequence, Tuple

import torch 
from torch import Tensor

from .bridge import Bridge
from ..debiasing.importance_sampling import estimate_ess_ratio
from ..preconditioners import Preconditioner
from ..target_functions import RareEventFunc
from ..tools import compute_f_divergence


class SmoothedIndicator(Bridge, abc.ABC):
    
    def __init__(
        self, 
        gammas: Sequence | Tensor | float, 
        betas: Sequence | Tensor | float = 1.0
    ):
        self.gammas, self.betas = self._parse_bridging_params(gammas, betas)
        self.num_layers = 0
        self.initialised = False
        self.is_adaptive = False

        self._ratio_weight_funcs = {
            "aratio": self._compute_weights_aratio,
            "eratio": self._compute_weights_eratio
        }
        
        return
    
    @property
    def is_last(self) -> bool:
        return self.num_layers == (len(self.betas) - 1)

    @staticmethod
    def _parse_bridging_params(
        gammas, 
        betas
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        # TODO: this could be tidied up.

        if isinstance(gammas, Tensor):
            gammas = gammas.tolist()
        if isinstance(betas, Tensor):
            betas = betas.tolist()
        
        if isinstance(gammas, float):
            gammas = [gammas]
        if isinstance(betas, float):
            betas = [betas]

        if abs(betas[-1] - 1.0) > 1e-6:
            raise Exception("Final beta value must be equal to 1.")

        if len(gammas) == 1:
            gammas *= len(betas)
        if len(betas) == 1:
            betas *= len(gammas)

        betas = {k: beta for k, beta in enumerate(betas)}
        gammas = {k: gamma for k, gamma in enumerate(gammas)}
        betas[-1] = 0.0
        gammas[-1] = 0.0
        
        return gammas, betas
    
    @abc.abstractmethod
    def neglogsmoothind(self, gamma: float, responses: Tensor) -> Tensor:
        """Evaluates the negative logarithm of the smooth surrogate to 
        the indicator function for a given value of the gamma parameter 
        and a set of response values.
        
        Parameters
        ----------
        gamma:
            The value of the shape parameter, gamma.
        responses:
            An n-dimensional vector containing the value of the 
            response function at each of the set of samples.
        
        Returns
        -------
        negloginds:
            An n-dimensional vector containing the negative logarithm 
            of the smooth surrogate to the indicator function evaluated 
            at each element in the response vector.

        """
        pass

    @abc.abstractmethod
    def grad_neglogsmoothind(self, gamma: float, responses: Tensor) -> Tensor:
        """TODO: write docstring for me..."""
        pass

    def reset(self) -> None:
        self.num_layers = 0
        return

    def initialise(
        self, 
        preconditioner: Preconditioner, 
        target_func: RareEventFunc
    ) -> None:
        
        if not isinstance(target_func, RareEventFunc):
            msg = "Target function must be of type 'RareEventFunc'."
            raise Exception(msg)

        Bridge.initialise(self, preconditioner, target_func)
        self.initialised = True
        return
    
    def _eval_pullback_split(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the pullback of the target density under the 
        preconditioning mapping.
        """
        xs, neglogdets = self.apply_preconditioner(us)
        neglogfxs, responses = self.target_func.func(xs)
        neglogfus = neglogfxs + neglogdets
        return neglogfus, responses
    
    def _eval_pullback_grad_split(
        self, 
        us: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        if not self.target_func.has_grad:
            msg = "Gradients of the target function have not been supplied."
            raise Exception(msg)

        self.target_func: RareEventFunc

        xs, neglogdets = self.apply_preconditioner(us)
        neglogfxs, grad_neglogfxs, responses, grad_responses = self.target_func.grad_func(xs)
        neglogfus = neglogfxs + neglogdets

        grad_neglogfus = grad_neglogfxs.clone()  # TODO: do this properly..

        return neglogfus, grad_neglogfus, responses, grad_responses

    def _compute_neglogbridges(
        self, 
        neglogref_us: Tensor,
        neglogfus: Tensor,
        responses: Tensor
    ) -> Tensor:
        
        k = self.num_layers
        neglogsigmoids_p = self.neglogsmoothind(self.gammas[k-1], responses)
        neglogbridges = (
            + (1.0 - self.betas[k-1]) * neglogref_us 
            + self.betas[k-1] * neglogfus 
            + neglogsigmoids_p
        )
        return neglogbridges

    def _compute_weights_aratio(
        self,
        neglogref_us: Tensor, 
        neglogfus: Tensor, 
        responses: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        """Computes the ratio between the current bridging density and 
        the previous bridging density for each particle.
        """
        
        k = self.num_layers
        negloginds = self.neglogsmoothind(self.gammas[k], responses)
        negloginds_p = self.neglogsmoothind(self.gammas[k-1], responses)
        negloginds_p[negloginds_p.isinf()] = 0.0
        
        neglogweights = (
            + (self.betas[k-1] - self.betas[k]) * neglogref_us 
            + (self.betas[k] - self.betas[k-1]) * neglogfus 
            + (negloginds - negloginds_p)
        )
        return neglogweights

    def _compute_weights_eratio(
        self,
        neglogref_us, 
        neglogfus, 
        responses,
        neglogfus_dirt
    ) -> Tensor:
        """Computes the ratio between the current bridging density and 
        the DIRT approximation to the previous bridging density for 
        each particle.
        """

        k = self.num_layers
        negloginds = self.neglogsmoothind(self.gammas[k], responses)
        
        neglogweights = (
            + (1.0 - self.betas[k]) * neglogref_us 
            + self.betas[k] * neglogfus
            + negloginds
            - neglogfus_dirt
        )
        return neglogweights
    
    def _compute_ratio_func(
        self, 
        method: str,
        neglogref_rs: Tensor,
        neglogref_us: Tensor, 
        neglogfus: Tensor, 
        responses: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:

        neglogratios = self._ratio_weight_funcs[method](
            neglogref_us,
            neglogfus, 
            responses, 
            neglogfus_dirt
        ) + neglogref_rs
        return neglogratios
    
    def _compute_log_weights(
        self,
        neglogref_us: Tensor,
        neglogfus: Tensor, 
        responses: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tensor:
        """Returns the logarithm of the ratio between the next bridging 
        density and the current bridging density.
        """
        neglogweights = self._ratio_weight_funcs["aratio"](
            neglogref_us,
            neglogfus, 
            responses, 
            neglogfus_dirt
        )
        return -neglogweights
    
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
        neglogfus, responses = self._eval_pullback_split(us)

        neglogratios = self._compute_ratio_func(
            method,
            neglogref_rs,
            neglogref_us,
            neglogfus,
            responses,
            neglogfus_dirt
        )
        return neglogratios

    def update(self, us: Tensor, neglogfus_dirt: Tensor) -> Tuple[Tensor, Tensor]:
        
        if not self.initialised:
            raise Exception("Need to call self.initialise().")
        
        neglogref_us = self.reference.eval_potential(us)[0]
        neglogfus, responses = self._eval_pullback_split(us)

        neglogbridges = self._compute_neglogbridges(
            neglogref_us,
            neglogfus,
            responses
        )

        log_weights = self._compute_log_weights(
            neglogref_us,
            neglogfus, 
            responses, 
            neglogfus_dirt
        )

        return log_weights, neglogbridges
    
    def eval_gradneglog(self, us) -> Tuple[Tensor, Tensor]:
        """Evaluates the current bridging density and its gradient at a 
        set of samples.
        """

        neglogref_us, grad_neglogref_us = self.reference.eval_potential(us)
        neglogfus, grad_neglogfus, responses, grad_responses = self._eval_pullback_grad_split(us)

        k = self.num_layers
        neglogsigmoids, grad_neglogsigmoids = self.grad_neglogsmoothind(self.gammas[k], responses)

        grad_neglogsigmoids = grad_neglogsigmoids[:, None] * grad_responses # TODO: this gradient is in terms of x, rather than u
        grad_neglogsigmoids[grad_neglogsigmoids.isnan()] = 0.0 # TODO: figure out what the correct value is here.

        neglogbridges = (
            (1.0 - self.betas[k]) * neglogref_us 
            + self.betas[k] * neglogfus 
            + neglogsigmoids
        )

        # TODO: finite difference check on these.
        grad_neglogbridges = (
            (1.0 - self.betas[k]) * grad_neglogref_us
            + self.betas[k] * grad_neglogfus
            + grad_neglogsigmoids
        )

        return neglogbridges, grad_neglogbridges

    def _get_diagnostics(
        self, 
        log_weights: Tensor | None,
        neglogfus: Tensor | None,
        neglogfus_dirt: Tensor | None
    ) -> List[str]:
        
        msg = [
            f"Gamma: {self.gammas[self.num_layers]:.4f}",
            f"Beta: {self.betas[self.num_layers]:.4f}"
        ]
        
        if (isinstance(log_weights, NoneType) 
            or isinstance(neglogfus, NoneType)
            or isinstance(neglogfus_dirt, NoneType)): 
            return msg

        div_h2 = compute_f_divergence(-neglogfus_dirt, -neglogfus)
        ess = estimate_ess_ratio(log_weights)
        msg += [f"DHell: {div_h2.sqrt():.4f}", f"ESS: {ess:.4f}"]
        return msg


class SigmoidSmoothing(SmoothedIndicator):
    r"""Uses a sigmoid function in place of an indicator function.

    This bridge must be used with a `RareEventFunc` as the target 
    function.

    Parameters
    ----------
    gammas:
        A sequence of values, $\{\gamma_{k}\}_{k=1}^{N}$, which define 
        the sigmoid functions.
    betas:
        A sequence of values, $\{\beta_{k}\}_{k=1}^{N}$, to use to 
        temper the density of the parameter. If these are not provided, 
        a value of $\beta_{k}=1$ will be used when defining all 
        intermediate densities.

    Notes
    -----
    This bridge is used in rare event estimation problems to 
    approximate the optimal biasing density, which takes the form
    $$
        \pi^{*}(\theta) \propto \pi(\theta)\mathbb{I}_{\mathcal{F}}(\theta), 
        \qquad \textrm{where } \mathcal{F} := \{\theta : F(\theta) \geq z\}.
    $$
    In the above, $\theta$ denotes a set of parameters with density 
    $\pi(\cdot)$, $F(\cdot)$ denotes the system response function, and 
    $z$ denotes a (scalar--valued) rare event threshold.
    
    The intermediate densities generated using this approach take the 
    form [@Cui2023]
    $$
        \pi_{k}(\theta) \propto (Q_{\sharp}\rho(\theta))^{1-\beta_{k}}
            \pi(\theta)^{\beta_{k}}g_{\gamma_{k}}(z).
    $$
    In the above, $Q_{\sharp}\rho(\cdot)$ denotes the pushforward of 
    the reference density, $\rho(\cdot)$, under the preconditioner, 
    $Q(\cdot)$, and $g_{\gamma_{k}}(\cdot)$ denotes the sigmoid 
    function, which is defined as
    $$
        g_{\gamma_{k}}(z) := (1 + \exp(\gamma_{k}(F(\theta) - z)))^{-1}.
    $$
    The sequences $\{\beta_{k}\}_{k=1}^{N}$ and 
    $\{\gamma_{k}\}_{k=1}^{N}$ must satisfy 
    $0 \leq \gamma_{1} \leq \cdots \leq \gamma_{N}$ and
    $0 \leq \beta_{1} \leq \cdots \leq \beta_{N} = 1$.

    """
    
    def neglogsmoothind(self, gamma: float, Fs: Tensor) -> Tensor:
        lsfs = self.target_func.threshold - Fs  # type: ignore
        neglogsigmoids = torch.log1p(torch.exp(gamma * lsfs))
        return neglogsigmoids
    
    def grad_neglogsmoothind(self, gamma: float, Fs: Tensor) -> Tuple[Tensor, Tensor]:
        neglogsigmoids = self.neglogsmoothind(gamma, Fs)
        negloggrads = (
            - math.log(gamma)
            - gamma * (self.target_func.threshold - Fs)
            + 2.0 * neglogsigmoids
        )
        grad_neglogsigmoids = -torch.exp(-negloggrads+neglogsigmoids)
        return neglogsigmoids, grad_neglogsigmoids
    

class GaussianSmoothing(SmoothedIndicator):
    """Uses a Gaussian CDF in place of an indicator function.

    TODO: finish this docstring.
    """

    def neglogsmoothind(self, gamma: float, responses: Tensor) -> Tensor:
        lsfs = self.target_func.threshold - responses  # type: ignore
        neglogtanhs = math.log(2.0) - torch.log1p(torch.erf(-gamma*lsfs))
        return neglogtanhs
    
    def grad_neglogsmoothind(self, gamma: float, Fs: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()