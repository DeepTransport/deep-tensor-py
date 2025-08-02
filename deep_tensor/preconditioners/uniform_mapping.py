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

    def __init__(
        self, 
        bounds: Tensor, 
        reference: Reference | None = None
    ):
        
        bounds = torch.atleast_2d(bounds)
        if bounds.shape[1] != 2:
            msg = "Bounds array must have two columns."
            raise Exception(msg)

        if reference is None:
            reference = GaussianReference()
        
        self.lbs, self.ubs = bounds.T
        self.dxs = self.ubs - self.lbs
        self.reference = reference
        self.dim = bounds.shape[0]
        return

    def Q(self, us: Tensor, subset: str = "first") -> Tensor:
        # Reference to uniform
        d_us = us.shape[1]
        zs = self.reference.eval_cdf(us)[0]
        if subset == "first":
            xs = self.lbs[:d_us] + self.dxs[:d_us] * zs 
        elif subset == "last":
            xs = self.lbs[-d_us:] + self.dxs[-d_us:] * zs
        return xs 
    
    def Q_inv(self, xs: Tensor, subset: str = "first") -> Tensor:
        # Uniform to reference
        d_xs = xs.shape[1]
        if subset == "first":
            zs = (xs - self.lbs[:d_xs]) / self.dxs[:d_xs]
        elif subset == "last":
            zs = (xs - self.lbs[-d_xs:]) / self.dxs[-d_xs:]
        us = self.reference.invert_cdf(zs)
        return us
    
    def neglogdet_Q(self, us: Tensor, subset: str = "first") -> Tensor:
        n_us, d_us = us.shape
        if subset == "first":
            neglogfxs = self.dxs[:d_us].log().sum().item()
        elif subset == "last":
            neglogfxs = self.dxs[-d_us:].log().sum().item()
        neglogfxs = torch.full((n_us,), neglogfxs)
        return self.reference.eval_potential(us)[0] - neglogfxs
    
    def neglogdet_Q_inv(self, xs: Tensor, subset: str = "first") -> Tensor:
        n_xs, d_xs = xs.shape
        if subset == "first":
            neglogfxs = self.dxs[:d_xs].log().sum().item()
        elif subset == "last":
            neglogfxs = self.dxs[-d_xs:].log().sum().item()
        neglogfxs = torch.full((n_xs,), neglogfxs)
        us = self.Q_inv(xs, subset)
        return neglogfxs - self.reference.eval_potential(us)[0]