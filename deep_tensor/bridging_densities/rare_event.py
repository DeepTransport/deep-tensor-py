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
        self._eval_neglogratio_funcs = {
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
    def neglogsmoothind(self, gamma: float, Fs: Tensor) -> Tensor:
        """Evaluates the negative logarithm of the smooth surrogate to 
        the indicator function for a given value of the gamma parameter 
        and a set of response values.
        
        Parameters
        ----------
        gamma:
            The value of the shape parameter, gamma.
        Fs:
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
    def grad_neglogsmoothind(
        self, 
        gamma: float, 
        Fs: Tensor
    ) -> Tuple[Tensor, Tensor]:
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
        xs, neglogdets = self.preconditioner.Q(us)
        neglogfxs, Fs = self.target_func.func(xs)
        neglogfus = neglogfxs + neglogdets
        return neglogfus, Fs
    
    def _grad_pullback_split(
        self, 
        us: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        self._check_grad()
        self.target_func: RareEventFunc

        xs, neglogdets, dxdus = self.preconditioner.grad_Q(us)
        neglogfxs, grad_neglogfxs, Fs, dFdxs = self.target_func.grad_func(xs)
        neglogfus = neglogfxs + neglogdets
        
        grad_neglogfus = self._grad_chain(grad_neglogfxs, dxdus)
        dFdus = self._grad_chain(dFdxs, dxdus)

        return neglogfus, grad_neglogfus, Fs, dFdus

    def _eval_neglogweights_aratio(
        self,
        neglogref_us: Tensor, 
        neglogfus: Tensor, 
        Fs: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        """Computes the ratio between the current bridging density and 
        the previous bridging density for each particle.
        """
        
        k = self.num_layers
        negloginds = self.neglogsmoothind(self.gammas[k], Fs)
        negloginds_p = self.neglogsmoothind(self.gammas[k-1], Fs)
        negloginds_p[negloginds_p.isinf()] = 0.0
        
        neglogweights = (
            + (self.betas[k-1] - self.betas[k]) * neglogref_us 
            + (self.betas[k] - self.betas[k-1]) * neglogfus 
            + (negloginds - negloginds_p)
        )
        return neglogweights
    
    def _grad_neglogweights_aratio(
        self,
        neglogref_us: Tensor,
        grad_neglogref_us: Tensor,
        neglogfus: Tensor, 
        grad_neglogfus: Tensor,
        Fs: Tensor, 
        dFdus: Tensor,
        neglogfus_dirt: Tensor,
        grad_neglogfus_dirt: Tensor | None
    ) -> Tuple[Tensor, Tensor]:
        k = self.num_layers
        neglogweights = self._eval_neglogweights_aratio(
            neglogref_us, 
            neglogfus, 
            Fs, 
            neglogfus_dirt
        )
        # Compute gradient of indicator function w.r.t. u
        # TODO: wrap this into its own function.
        # TODO: figure out whether I need to do any post-processing here..
        grad_negloginds = self.grad_neglogsmoothind(self.gammas[k], Fs)[1]
        grad_negloginds_p = self.grad_neglogsmoothind(self.gammas[k-1], Fs)[1]
        # grad_negloginds = self.grad_neglogsmoothind(self.gammas[k], Fs)[1]
        grad_negloginds = grad_negloginds[:, None] * dFdus
        grad_negloginds_p = grad_negloginds_p[:, None] * dFdus

        grad_neglogweights = (
            + (self.betas[k-1] - self.betas[k]) * grad_neglogref_us 
            + (self.betas[k] - self.betas[k-1]) * grad_neglogfus 
            + (grad_negloginds - grad_negloginds_p)
        )
        return neglogweights, grad_neglogweights
    
    def _eval_neglogweights_eratio(
        self,
        neglogref_us, 
        neglogfus, 
        Fs,
        neglogfus_dirt
    ) -> Tensor:
        """Computes the negative logarithm of the ratio between the 
        current bridging density and the DIRT approximation to the 
        previous bridging density for each particle.
        """
        k = self.num_layers
        negloginds = self.neglogsmoothind(self.gammas[k], Fs)
        neglogweights = (
            + (1.0 - self.betas[k]) * neglogref_us 
            + self.betas[k] * neglogfus
            + negloginds
            - neglogfus_dirt
        )
        return neglogweights

    def _grad_neglogweights_eratio(
        self,
        neglogref_us: Tensor,
        grad_neglogref_us: Tensor,
        neglogfus: Tensor, 
        grad_neglogfus: Tensor,
        Fs: Tensor, 
        dFdus: Tensor,
        neglogfus_dirt: Tensor,
        grad_neglogfus_dirt: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        neglogweights = self._eval_neglogweights_eratio(
            neglogref_us,
            neglogfus,
            Fs,
            neglogfus_dirt
        )

        k = self.num_layers
        grad_negloginds = self.grad_neglogsmoothind(self.gammas[k], Fs)[1]
        grad_negloginds = grad_negloginds[:, None] * dFdus
        
        grad_neglogweights = (
            + (1.0 - self.betas[k]) * grad_neglogref_us 
            + self.betas[k] * grad_neglogfus
            + grad_negloginds
            - grad_neglogfus_dirt
        )
        return neglogweights, grad_neglogweights
    
    def _eval_neglogratio(
        self,
        method: str,
        rs: Tensor,
        us: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        
        # TODO: wrap this into a function.
        if not self.initialised:
            raise Exception("Need to call self.initialise().")
        
        neglogref_rs = self.reference.eval_potential_unnormalised(rs)[0]
        neglogref_us = self.reference.eval_potential_unnormalised(us)[0]
        neglogfus, Fs = self._eval_pullback_split(us)

        neglogratios = self._eval_neglogratio_funcs[method](
            neglogref_us,
            neglogfus, 
            Fs, 
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

        # TODO: finite difference check on this output!!
        
        # TODO: the naming for eval_potential could be split into 
        # a function of the same name and grad_potential... it probably
        # doesn't matter that much though

        neglogref_rs, grad_neglogref_rs = self.reference.eval_potential_unnormalised(rs)
        neglogref_us, grad_neglogref_us = self.reference.eval_potential_unnormalised(us)

        neglogfus, grad_neglogfus, Fs, dFdus = self._grad_pullback_split(us)

        neglogweights, grad_neglogweights = self._grad_neglogweight_funcs[method](
            neglogref_us,
            grad_neglogref_us,
            neglogfus, 
            grad_neglogfus,
            Fs, 
            dFdus,
            neglogfus_dirt,
            grad_neglogfus_dirt
        )
        # Convert gradient of log-weights w.r.t. u to gradient w.r.t. r
        grad_neglogweights = self._grad_chain(grad_neglogweights, dudrs)

        neglogratios = neglogweights + neglogref_rs
        grad_neglogratios = grad_neglogweights + grad_neglogref_rs
        return neglogratios, grad_neglogratios
    
    def _eval_neglogbridge(
        self, 
        neglogref_us: Tensor,
        neglogfus: Tensor,
        Fs: Tensor,
        num_layers: int | None = None  # in case we want to evaluate a previous density
    ) -> Tensor:
        k = num_layers if num_layers is not None else self.num_layers
        neglogsigmoids = self.neglogsmoothind(self.gammas[k], Fs)
        neglogbridges = (
            + (1.0 - self.betas[k]) * neglogref_us 
            + self.betas[k] * neglogfus 
            + neglogsigmoids
        )
        return neglogbridges

    def _grad_neglogbridge(
        self, 
        us: Tensor, 
        dudrs: Tensor 
    ) -> Tuple[Tensor, Tensor]:

        # TODO: finite difference check on this output!!

        neglogref_us, grad_neglogref_us = self.reference.eval_potential_unnormalised(us)
        neglogfus, grad_neglogfus, Fs, dFdus = self._grad_pullback_split(us)

        k = self.num_layers
        gamma, beta = self.gammas[k], self.betas[k]
        grad_negloginds = self.grad_neglogsmoothind(gamma, Fs)[1]
        grad_negloginds = grad_negloginds[:, None] * dFdus
        # TODO: figure out what the correct value is here. 
        # also probably a good idea to add this to the logging output.
        grad_negloginds[grad_negloginds.isnan()] = 0.0

        neglogbridges = self._eval_neglogbridge(neglogref_us, neglogfus, Fs)

        # Compute gradient w.r.t. u
        grad_neglogbridges = (
            (1.0 - beta) * grad_neglogref_us
            + beta * grad_neglogfus
            + grad_negloginds
        )
        grad_neglogbridges = torch.nan_to_num(grad_neglogbridges)
        # Change variable such that gradient is w.r.t. r
        grad_neglogbridges = self._grad_chain(grad_neglogbridges, dudrs)

        return neglogbridges, grad_neglogbridges

    def _compute_log_weights(
        self,
        neglogref_us: Tensor,
        neglogfus: Tensor, 
        Fs: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tensor:
        """Returns the logarithm of the ratio between the next bridging 
        density and the current bridging density.
        """
        neglogweights = self._eval_neglogratio_funcs["aratio"](
            neglogref_us,
            neglogfus, 
            Fs, 
            neglogfus_dirt
        )
        return -neglogweights

    def update(self, us: Tensor, neglogfus_dirt: Tensor) -> Tuple[Tensor, Tensor]:
        
        if not self.initialised:
            raise Exception("Need to call self.initialise().")
        
        neglogref_us = self.reference.eval_potential_unnormalised(us)[0]
        neglogfus, Fs = self._eval_pullback_split(us)

        neglogbridges = self._eval_neglogbridge(
            neglogref_us,
            neglogfus,
            Fs,
            num_layers=self.num_layers-1
        )

        log_weights = self._compute_log_weights(
            neglogref_us,
            neglogfus, 
            Fs, 
            neglogfus_dirt
        )

        return log_weights, neglogbridges

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
            - torch.tensor(gamma).log()
            - gamma * (self.target_func.threshold - Fs)
            + 2.0 * neglogsigmoids
        )
        grad_neglogsigmoids = -torch.exp(-negloggrads+neglogsigmoids)
        return neglogsigmoids, grad_neglogsigmoids
    

class GaussianSmoothing(SmoothedIndicator):
    """Uses a Gaussian CDF in place of an indicator function.

    TODO: finish this docstring.
    """

    def neglogsmoothind(self, gamma: float, Fs: Tensor) -> Tensor:
        lsfs = self.target_func.threshold - Fs  # type: ignore
        neglogtanhs = math.log(2.0) - torch.log1p(torch.erf(-gamma*lsfs))
        return neglogtanhs
    
    def grad_neglogsmoothind(self, gamma: float, Fs: Tensor) -> Tuple[Tensor, Tensor]:
        lsfs = self.target_func.threshold - Fs
        neglogpdfs = lsfs**2 * gamma**2 + 0.5*torch.log(torch.pi / (torch.tensor(gamma)**2))
        neglogcdfs = self.neglogsmoothind(gamma, Fs)
        grad_neglogcdfs = -torch.exp(neglogcdfs - neglogpdfs)
        return neglogcdfs, grad_neglogcdfs