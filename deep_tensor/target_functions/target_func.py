from typing import Callable
import warnings

import torch
from torch import Tensor


class TargetFunc(object):
    r"""An arbitrary target density function to be approximated.
    
    Parameters
    ----------
    neglogfx:
        A function which takes an $n \times d$ matrix in which each row 
        contains a sample of the parameters, and returns an 
        $n$-dimensional vector which contains the negative logarithm of 
        (a possibly unnormalised version of) the target density.
        
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
            msg = "Target function takes values of infinity."
            warnings.warn(msg)
        return neglogfxs