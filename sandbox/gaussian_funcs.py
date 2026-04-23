import math

from torch import Tensor


def eval_unit_gaussian_normalised(xs: Tensor) -> Tensor:
    """Evaluates the negative logarithm of the normalised unit normal 
    density.
    
    Parameters
    ----------
    xs:
        An n * d matrix containing a set of points at which to evaluate 
        the density.

    Returns
    -------
    neglogfxs:
        An n-dimensional vector containing the potential function 
        associated with the (normalised) target density evaluated at 
        each sample in xs. 
    
    """
    dim = xs.shape[1]
    neglognorm = 0.5 * dim * math.log(2.0 * math.pi)
    return 0.5 * xs.square().sum(dim=1) + neglognorm


def eval_unit_gaussian_unnormalised(xs: Tensor) -> Tensor:
    """Evaluates the negative logarithm of the unnormalised unit normal 
    density.
    
    Parameters
    ----------
    xs:
        An n * d matrix containing a set of points at which to evaluate 
        the density.

    Returns
    -------
    neglogfxs:
        An n-dimensional vector containing the potential function 
        associated with the (unnormalised) target density evaluated at 
        each sample in xs. 
    
    """
    return 0.5 * xs.square().sum(dim=1)