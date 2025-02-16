from typing import Callable

import torch
from torch import Tensor


def integrate(
    func: Callable[[Tensor], Tensor], 
    x0: float,
    x1: float,
    n: int=151
) -> Tensor:
    """Approximates the integral of a given function on the interval 
    [x0, x1] using the trapezoidal rule.

    Parameters
    ----------
    func:
        TODO: write this. Note that func should be vectorised.

    TODO: could make this adaptive in future.
    """
    
    xs = torch.linspace(x0, x1, n)
    ys = func(xs)
    return torch.trapezoid(ys, xs)