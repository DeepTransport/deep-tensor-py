from typing import Tuple

import torch
from torch import Tensor
from torch import linalg

from .preconditioner import Preconditioner
from ..references import GaussianReference


class AffineMapping(Preconditioner):
    r"""An affine transformation.

    Parameters
    ----------
    A:
        A non-singular n*n matrix.
    b:
        An n-dimensional vector. If not provided, this will be set to 
        zero.
    reference:
        The reference density. If this is not specified, it will be set 
        to the unit Gaussian density with support on $[-4, 4]^{d}$.

    """

    def __init__(
        self,
        A: Tensor,
        b: Tensor | None = None, 
        reference: GaussianReference | None = None
    ):
        if b is None:
            b = torch.zeros((A.shape[0],))
        if reference is None:
            reference = GaussianReference()
        self.A = A
        self.b = b.flatten()
        self.reference = reference
        self.A_inv: Tensor = linalg.inv(self.A)
        self.dim = self.b.flatten().numel()
        return

    def _check_shape(self, xs: Tensor) -> None:
        if xs.shape[1] != self.dim:
            msg = ("Preconditioner is not defined for a subset of "
                   "the variables.")
            raise Exception(msg)
        return

    def Q(self, us: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor]:
        self._check_shape(us)
        xs = self.b + us @ self.A.T
        neglogdet = -self.A.slogdet().logabsdet.item()
        neglogdets = torch.full((us.shape[0],), neglogdet, device=us.device)
        return xs, neglogdets
    
    def Q_inv(self, xs: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor]:
        self._check_shape(xs)
        us = (xs - self.b) @ self.A_inv.T
        neglogdet = -self.A_inv.slogdet().logabsdet.item()
        neglogdets = torch.full((xs.shape[0],), neglogdet, device=xs.device)
        return us, neglogdets

    def grad_Q(self, us: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor, Tensor]:
        self._check_shape(us)
        num_us = us.shape[0]
        xs, neglogdets = self.Q(us)
        dxdus = self.A[:, None, :].repeat(1, num_us, 1)
        return xs, neglogdets, dxdus