import torch
from torch import Tensor

from .preconditioner import Preconditioner
from ..references import GaussianReference, Reference


class UniformMapping(Preconditioner):
    r"""A mapping between the reference density and a uniform density.

    The uniform density can have an arbitrary set of bounds in each 
    dimension.

    This preconditioner is diagonal.
    
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
        dxs = ubs - lbs

        def Q(us: Tensor) -> Tensor:
            # Reference to uniform
            d_us = us.shape[1]
            zs = reference.eval_cdf(us)[0]
            xs = lbs[:d_us] + dxs[:d_us] * zs 
            return xs 
        
        def Q_inv(xs: Tensor) -> Tensor:
            # Uniform to reference
            d_xs = xs.shape[1]
            zs = (xs - lbs[:d_xs]) / dxs[:d_xs]
            us = reference.invert_cdf(zs)
            return us
        
        def neglogdet_Q(us: Tensor) -> Tensor:
            n_us, d_us = us.shape
            neglogfxs = torch.full((n_us,), dxs[:d_us].prod().log())
            return reference.eval_potential(us)[0] - neglogfxs
        
        def neglogdet_Q_inv(xs: Tensor) -> Tensor:
            n_xs, d_xs = xs.shape
            us = Q_inv(xs)
            neglogfxs = torch.full((n_xs,), dxs[:d_xs].prod().log())
            return neglogfxs - reference.eval_potential(us)[0]

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