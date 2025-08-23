from typing import Callable, Tuple
import warnings

import torch
from torch import Tensor

from .target_func import TargetFunc


class RareEventFunc(TargetFunc):
    r"""A function for rare event estimation problems.
    
    Parameters
    ----------
    func:
        A function which takes an $n \times d$ matrix in which each row 
        contains a sample of the parameters, and returns an 
        $n$-dimensional which contains the negative logarithm of 
        (a possibly unnormalised version of) the target density, and an 
        $n$-dimensional vector containing the response function 
        evaluated at each sample.
    threshold: 
        The threshold, $z$, which defines a rare event.

    Notes
    -----
    This target function is used for problems in which we have a set of 
    parameters, $\theta \in \mathbb{R}^{n}$, with density $\pi(\cdot)$, 
    and want to estimate the probability that some response function 
    $F : \mathbb{R}^{n} \rightarrow \mathbb{R}$ is greater than or 
    equal to a threshold, $z$; that is,
    $$
        \mathbb{E}_{\pi}[\mathbb{I}_{\mathcal{F}}(\theta)], 
    $$
    where $\mathbb{I}_{\mathcal{F}}$ denotes the indicator function of 
    the set $\mathcal{F}$, which is defined as 
    $$
        \mathcal{F} := \{\theta : F(\theta) \geq z\}.
    $$ 
        
    """

    def __init__(
        self, 
        func: Callable[[Tensor], Tuple[Tensor, Tensor]],
        threshold: float,
        vectorised: bool = True
    ):
        self._func = func
        self.threshold = threshold
        self.vectorised = vectorised
        return
    
    def __call__(self, xs: Tensor) -> Tensor:
        """Returns the negative logarithm of the product of (a quantity 
        proportional to) the density of the parameters and the rare 
        event indicator function.
        """
        neglogfxs, responses = self.func(xs)
        rare_event_indicator = responses > self.threshold
        neglogfxs[~rare_event_indicator] = torch.inf
        return neglogfxs
    
    def _func_vectorised(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        
        if self.vectorised:
            return self._func(xs)
        
        neglogfxs = torch.zeros((xs.shape[0],))
        responses = torch.zeros((xs.shape[0],))
        for i, x in enumerate(xs.T):
            neglogfxs[i], responses[i] = self._func(x)
        return neglogfxs, responses
    
    def func(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        neglogfxs, responses = self._func_vectorised(xs)
        num_infs = torch.sum(neglogfxs == -torch.inf)
        if num_infs > 0:
            msg = "Target function takes values of infinity."
            warnings.warn(msg)
        return neglogfxs, responses