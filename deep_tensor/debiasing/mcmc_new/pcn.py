import math
import random
from typing import Callable, Tuple
import warnings

import torch 
from torch import Tensor

from .kernel import Kernel
from ...irt import DIRT
from ...references import GaussianReference


class pCNKernel(Kernel):

    def __init__(
        self, 
        potential: Callable[[Tensor], Tensor], 
        dirt: DIRT, 
        y: Tensor | None = None,
        subset: str = "first",
        dt: float = 10.0
    ):
        
        if not isinstance(dirt.reference, GaussianReference):
            msg = "pCN kernel requires a Gaussian reference."
            raise Exception(msg)
        
        if dt <= 0.0:
            msg = "Stepsize must be positive."
            raise Exception(msg)
        
        if dt == 2.0:
            msg = (
                "Setting dt=2.0 in the pCN kernel results in an " 
                "independence sampler. It is probably more efficient "
                "to use the dedicated independence sampling function."
            )
            warnings.warn(msg)

        self.a = 2.0 * math.sqrt(2.0*dt) / (2.0+dt)
        self.b = (2.0-dt) / (2.0+dt)

        Kernel.__init__(self, potential, dirt, y, subset)
        return
    
    def _step(self) -> Tuple[Tensor, Tensor]:
        """Takes a single pCN step."""

        if not self.initialised:
            msg = "Kernel not initialised."
            raise Exception(msg)

        xi = torch.randn((1, self.dim))

        r_prop = self.b * self._r_prev + self.a * xi

        if self._out_domain(r_prop):
            return self._x_prev, self._neglogfx_prev

        # Evaluate the potential of the pullback of the likelihood 
        # function evaluated at the proposed state
        neglogfr_prop, neglogfx_prop = self._potential_pull(r_prop)
        neglogref_prop = self.reference.eval_potential(r_prop)[0]
        negloglik_prop = neglogfr_prop - neglogref_prop

        alpha = self._negloglik_prev - negloglik_prop
        self.n_steps += 1

        if torch.exp(alpha) > random.random():
            self.n_accept += 1
            x_prop = self._irt_func(r_prop)
            self._r_prev = r_prop.clone()
            self._x_prev = x_prop.clone()
            self._negloglik_prev = negloglik_prop.clone()
            self._neglogfx_prev = neglogfx_prop.clone()
            return x_prop, neglogfx_prop
        
        return self._x_prev, self._neglogfx_prev