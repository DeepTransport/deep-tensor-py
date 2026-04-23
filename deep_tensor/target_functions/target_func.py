from typing import Callable, Tuple
import warnings

import torch
from torch import Tensor


class TargetFunc(object):
    r"""The negative logarithm of a density function to be approximated.
    
    Parameters
    ----------
    neglogfx:
        A function which returns the negative logarithm of a (possibly 
        unnormalised version of) the target density function. If 
        `vectorised=True`, the function should accept an $n \times d$ 
        matrix (where $n$ denotes the number of samples and $d$ denotes 
        the dimension of the parameters), and return an $n$-dimensional 
        vector containing the function evaluated at each sample. If 
        `vectorised=False`, the function should accept a $d$-dimensional 
        vector and return a single scalar value.
    grad_neglogfx:
        A function which returns the negative logarithm of a (possibly 
        unnormalised version of) the target density function, and the 
        gradient of the negative logarithm of the target density 
        function. The format of the arguments and returns is the same 
        as `neglogfx`.
    vectorised:
        Whether the `neglogfx` and `grad_neglogfx` accept multiple sets 
        of parameters.

    """

    def __init__(
        self, 
        neglogfx: Callable[[Tensor], Tensor],
        grad_neglogfx: Callable[[Tensor], Tuple[Tensor, Tensor]] | None = None,
        vectorised: bool = True
    ):
        self._func = neglogfx
        self._grad_func = grad_neglogfx
        self._is_vectorised = vectorised
        self._has_grad = self._grad_func is not None
        return
    
    def __call__(self, xs: Tensor) -> Tensor:
        """Syntax sugar for self.func()."""
        return self.func(xs)
    
    def _check_neglogfxs(self, neglogfxs: Tensor) -> None:
        """Checks whether any evaluations of the target function are 
        infinity.
        """
        num_infs = torch.sum(neglogfxs == -torch.inf)
        if num_infs > 0:
            msg = "Target function is not finite."
            warnings.warn(msg)
        return
    
    def _func_vectorised(self, xs: Tensor) -> Tensor:
        if self._is_vectorised:
            return self._func(xs)
        return torch.tensor([self._func(x) for x in xs], device=xs.device)
    
    def _grad_func_vectorised(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        
        if self._grad_func is None:
            msg = "No gradients of the target density have been provided."
            raise Exception(msg)
        
        if self._is_vectorised:
            return self._grad_func(xs)
        
        num_xs = xs.shape[0]
        neglogfxs = torch.zeros((num_xs,), device=xs.device)
        grad_neglogfxs = torch.zeros_like(xs)
        for i, x in enumerate(xs):
            neglogfxs[i], grad_neglogfxs[i] = self._grad_func(x)
        return neglogfxs, grad_neglogfxs

    def func(self, xs: Tensor) -> Tensor:
        """Evaluates the target function.
        
        Parameters
        ----------
        xs:
            An n * d matrix containing a set of samples at which to 
            evaluate the target function.
        
        Returns
        -------
        neglogfxs:
            An n-dimensional vector containing the target function 
            evaluated at each sample in `xs`.

        """
        neglogfxs = self._func_vectorised(xs)
        self._check_neglogfxs(neglogfxs)
        return neglogfxs
    
    def grad_func(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the target function and its gradient.
        
        Parameters
        ----------
        xs:
            An n * d matrix containing a set of samples at which to 
            evaluate the target function and its gradient.
        
        Returns
        -------
        neglogfxs:
            An n-dimensional vector containing the target function 
            evaluated at each sample in `xs`.
        grad_neglogfxs:
            An n * d matrix where each row contains the gradient of the 
            target function evaluated at sample in the corresponding 
            row of `xs`.

        """
        neglogfxs, grad_neglogfxs = self._grad_func_vectorised(xs)
        self._check_neglogfxs(neglogfxs)
        return neglogfxs, grad_neglogfxs