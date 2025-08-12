from typing import Callable

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

    def __init__(self, neglogfx: Callable[[Tensor], Tensor]):
        self.neglogfx = neglogfx
        return
    
    def __call__(self, xs: Tensor) -> Tensor:
        return self.neglogfx(xs)