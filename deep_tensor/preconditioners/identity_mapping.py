from typing import Tuple

import torch
from torch import Tensor

from .preconditioner import Preconditioner
from ..references import Reference, GaussianReference


class IdentityMapping(Preconditioner):
    r"""An identity mapping.

    This preconditioner is diagonal.

    Parameters
    ----------
    dim: 
        The dimension of the target (and reference) random variables.
    reference:
        The reference density. If this is not specified, it will 
        default to the unit Gaussian in $d$ dimensions with support 
        truncated to $[-4, 4]^{d}$.

    """

    def __init__(
        self, 
        dim: int, 
        reference: Reference | None = None
    ):
        self.dim = dim
        self.reference = GaussianReference() if reference is None else reference
        return

    @staticmethod
    def Q(us: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor]:
        num_us = us.shape[0]
        neglogdets = torch.zeros((num_us,), device=us.device)
        return us, neglogdets
    
    @staticmethod
    def Q_inv(xs: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor]:
        num_xs = xs.shape[0]
        neglogdets = torch.zeros((num_xs,), device=xs.device)
        return xs, neglogdets
    
    @staticmethod 
    def grad_Q(us: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor, Tensor]:
        num_us, dim_us = us.shape
        neglogdets = torch.zeros((num_us,), device=us.device)
        dxdus = torch.eye(dim_us, device=us.device)[:, None, :].repeat(1, num_us, 1)
        return us, neglogdets, dxdus