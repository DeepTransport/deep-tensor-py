from typing import Callable, Tuple

import torch
from torch import Tensor

from .target_func import TargetFunc


class RareEventFunc(TargetFunc):
    r"""A function for rare event estimation problems.
    
    Parameters
    ----------
    func:
        A function which returns the negative logarithm of a (possibly 
        unnormalised version of) the target density function, and the 
        response function. If `vectorised=True`, the function should 
        accept an $n \times d$ matrix (where $n$ denotes the number of 
        samples and $d$ denotes the dimension of the parameters), and 
        return an $n$-dimensional vector containing the negative 
        log-density function evaluated at each sample, and an 
        $n$-dimensional vector containing the response function 
        evaluated at each sample. If `vectorised=False`, the function 
        should accept a $d$-dimensional vector and return two scalar 
        values.
    threshold: 
        The threshold, $z$, which defines a rare event.
    grad_func:
        A function which returns the potential associated with the 
        target density function, the gradient of the potential with 
        respect to the parameters, the response function, and the 
        gradient of the response function with respect to the 
        parameters. The format of the arguments and returns is the same 
        as `func`.
    vectorised:
        Whether `func` and `grad_func` accept multiple sets of 
        parameters.

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
        grad_func: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]] | None = None,
        vectorised: bool = True
    ):
        self._func = func
        self._grad_func = grad_func
        self.threshold = threshold
        self._is_vectorised = vectorised
        self._has_grad = self._grad_func is not None
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
        if self._is_vectorised:
            return self._func(xs)
        num_xs = xs.shape[0]
        neglogfxs = torch.zeros((num_xs,), device=xs.device)
        responses = torch.zeros((num_xs,), device=xs.device)
        for i, x in enumerate(xs):
            neglogfxs[i], responses[i] = self._func(x)
        return neglogfxs, responses
    
    def _grad_func_vectorised(
        self, 
        xs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        if self._grad_func is None:
            msg = "No gradients of the biasing density have been provided."
            raise Exception(msg)

        if self._is_vectorised:
            return self._grad_func(xs)
        
        num_xs = xs.shape[0]
        neglogfxs = torch.zeros((num_xs,), device=xs.device)
        grad_neglogfxs = torch.zeros_like(xs)
        responses = torch.zeros((num_xs,), device=xs.device)
        grad_responses = torch.zeros_like(xs)
        
        for i, x in enumerate(xs):
            neglogfxs[i], grad_neglogfxs[i], responses[i], grad_responses[i] = self._grad_func(x)
        
        return neglogfxs, grad_neglogfxs, responses, grad_responses
    
    def func(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        neglogfxs, responses = self._func_vectorised(xs)
        self._check_neglogfxs(neglogfxs)
        return neglogfxs, responses
    
    def grad_func(
        self, 
        xs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        neglogfxs, grad_neglogfxs, responses, grad_responses = self._grad_func_vectorised(xs)
        self._check_neglogfxs(neglogfxs)
        return neglogfxs, grad_neglogfxs, responses, grad_responses