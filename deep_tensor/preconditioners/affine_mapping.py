from typing import Tuple

import torch
from torch import Tensor
from torch import linalg

from .preconditioner import Preconditioner
from ..references import GaussianReference


class AffineMapping(Preconditioner):
    r"""A mapping between two Gaussian densities.
    
    This preconditioner provides a mapping between the standard 
    Gaussian density and an Gaussian density with an arbitrary mean and 
    covariance.

    Parameters
    ----------
    mean:
        The mean of the target Gaussian density.
    cov:
        The covariance matrix of the target Gaussian density.
    reference:
        The reference density. If this is not specified, it will be set 
        to the unit Gaussian density with support on $[-4, 4]^{d}$.
    diag:
        Whether `cov` is a diagonal matrix.

    """

    def __init__(
        self,
        A: Tensor,
        b: Tensor | None = None, 
        reference: GaussianReference | None = None,
        diag: bool = False
    ):
        
        if b is None:
            b = torch.zeros((A.shape[0],))

        if reference is None:
            reference = GaussianReference()
        elif not isinstance(reference, GaussianReference):
            msg = "Reference density must be Gaussian."
            raise Exception(msg)

        self.A = A
        self.b = b
        self.reference = reference
        self.diag = diag
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