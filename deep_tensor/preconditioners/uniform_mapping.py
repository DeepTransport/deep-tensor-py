from typing import Tuple

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

    def Q(self, us: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor]:
        # Reference to uniform
        num_us, dim_us = us.shape
        zs = self.reference.eval_cdf(us)[0]
        if subset == "first":
            xs = self.lbs[:dim_us] + self.dxs[:dim_us] * zs 
            neglogfx = self.dxs[:dim_us].log().sum().item()
        elif subset == "last":
            xs = self.lbs[-dim_us:] + self.dxs[-dim_us:] * zs
            neglogfx = self.dxs[-dim_us:].log().sum().item()
        neglogfxs = torch.full((num_us,), neglogfx, device=us.device)
        neglogdets = self.reference.eval_potential(us)[0] - neglogfxs
        return xs, neglogdets
    
    def Q_inv(self, xs: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor]:
        # Uniform to reference
        num_xs, dim_xs = xs.shape
        if subset == "first":
            zs = (xs - self.lbs[:dim_xs]) / self.dxs[:dim_xs]
            neglogfx = self.dxs[:dim_xs].log().sum().item()
        elif subset == "last":
            zs = (xs - self.lbs[-dim_xs:]) / self.dxs[-dim_xs:]
            neglogfx = self.dxs[-dim_xs:].log().sum().item()
        us = self.reference.invert_cdf(zs)
        neglogfxs = torch.full((num_xs,), neglogfx, device=xs.device)
        neglogdets = neglogfxs - self.reference.eval_potential(us)[0]
        return us, neglogdets