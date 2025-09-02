from typing import Callable
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
    vectorised:
        Whether the function accepts multiple sets of parameters.

    """

    def __init__(
        self, 
        neglogfx: Callable[[Tensor], Tensor],
        vectorised: bool = True
    ):
        self._func = neglogfx
        self.vectorised = vectorised
        return
    
    def __call__(self, xs: Tensor) -> Tensor:
        return self.func(xs)
    
    def _func_vectorised(self, xs: Tensor) -> Tensor:
        if self.vectorised:
            return self._func(xs)
        return torch.tensor([self._func(x) for x in xs.T])
    
    def func(self, xs: Tensor) -> Tensor:
        neglogfxs = self._func_vectorised(xs)
        num_infs = torch.sum(neglogfxs == -torch.inf)
        if num_infs > 0:
            msg = "Target density is not finite."
            warnings.warn(msg)
        return neglogfxs