import torch
from torch import Tensor

from .preconditioner import Preconditioner
from ..references import GaussianReference, Reference


class UniformMapping(Preconditioner):
    r"""A mapping between an arbitrary (product form) reference density 
    and a uniform density with a specified set of bounds.
    
    Parameters
    ----------
    bounds:
        A $d \times 2$ matrix, where each row contains the lower and 
        upper bounds of the uniform density in each dimension.
    reference:
        The reference density. If this is not specified, it will 
        default to the unit Gaussian in $d$ dimensions with support 
        truncated to $[-4, 4]^{d}$.

    """

    def __init__(self, bounds: Tensor, reference: Reference | None = None):
        
        if reference is None:
            reference = GaussianReference()

        dim = bounds.shape[0]
        lbs, ubs = bounds.T
        dms = ubs - lbs

        def Q(xs: Tensor) -> Tensor:
            """Maps from reference to uniform."""
            d_xs = xs.shape[1]
            zs = reference.eval_cdf(xs)[0]
            ms = lbs[:d_xs] + dms[:d_xs] * zs 
            return ms 
        
        def Q_inv(ms: Tensor) -> Tensor:
            """Maps from uniform to reference."""
            d_ms = ms.shape[1]
            zs = (ms - lbs[:d_ms]) / dms[:d_ms]
            xs = reference.invert_cdf(zs)
            return xs
        
        def neglogdet_Q(xs: Tensor) -> Tensor:
            n_xs, d_xs = xs.shape
            neglogfms = torch.full((n_xs,), dms[:d_xs].prod().log())
            return reference.eval_potential(xs)[0] - neglogfms
        
        def neglogdet_Q_inv(ms: Tensor) -> Tensor:
            n_ms, d_ms = ms.shape
            xs = Q_inv(ms)
            neglogfms = torch.full((n_ms,), dms[:d_ms].prod().log())
            return neglogfms - reference.eval_potential(xs)[0]

        Preconditioner.__init__(
            self, 
            reference=reference,
            Q=Q, 
            Q_inv=Q_inv,
            neglogdet_Q=neglogdet_Q,
            neglogdet_Q_inv=neglogdet_Q_inv,
            dim=dim
        )

        return