from typing import Callable

import torch
from torch import Tensor


def integrate(
    func: Callable[[Tensor], Tensor], 
    x0: float | Tensor,
    x1: float | Tensor,
    n: int = 150
) -> Tensor:
    """Approximates the integral of a given function on the interval 
    [x0, x1] using the trapezoidal rule.

    Parameters
    ----------
    func:
        A function that takes in a vector of inputs and returns a 
        vector of the correpsonding values of the function.
    x0:
        Left-hand end of the integration interval.
    x1: 
        Right-hand end of the integration interval.
    n:
        The number of intervals to use to when applying the trapezoidal 
        rule.

    """
    xs = torch.linspace(x0, x1, n+1)
    ys = func(xs)
    return torch.trapezoid(ys, xs)