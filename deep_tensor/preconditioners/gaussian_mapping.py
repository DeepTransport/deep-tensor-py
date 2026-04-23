from typing import Tuple

import torch
from torch import Tensor
from torch import linalg

from .preconditioner import Preconditioner
from ..references import GaussianReference


class GaussianMapping(Preconditioner):
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
        mean: Tensor,
        cov: Tensor, 
        reference: GaussianReference | None = None,
        diag: bool = False
    ):
        if reference is None:
            reference = GaussianReference()
        elif not isinstance(reference, GaussianReference):
            msg = "Reference density must be Gaussian."
            raise Exception(msg)
        self.mean = mean.flatten()
        self.cov = cov 
        self.reference = reference
        self.diag = diag
        self.L: Tensor = linalg.cholesky(cov)
        self.R: Tensor = linalg.inv(self.L)
        self.dim = self.mean.flatten().numel()
        return

    def _check_subset(self, subset: str) -> None:
        if self.diag is False and subset == "last":
            msg = ("Preconditioner is only well-defined when "
                    "subset='first', unless diag=True.")
            raise Exception(msg)
        return

    def Q(self, us: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor]:
        self._check_subset(subset)
        dim_us = us.shape[1]
        if subset == "first":
            xs = self.mean[:dim_us] + (us @ self.L[:dim_us, :dim_us].T)
            Ls = self.L.diag()[:dim_us]
        else:
            xs = self.mean[-dim_us:] + (us @ self.L[-dim_us:, -dim_us:].T)
            Ls = self.L.diag()[-dim_us:]
        neglogdet = -Ls.log().sum().item()
        neglogdets = torch.full((us.shape[0],), neglogdet, device=us.device)
        return xs, neglogdets
    
    def Q_inv(self, xs: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor]:
        self._check_subset(subset)
        dim_xs = xs.shape[1]
        if subset == "first":
            us = (xs - self.mean[:dim_xs]) @ self.R[:dim_xs, :dim_xs].T
            Rs = self.R.diag()[:dim_xs]
        else:
            us = (xs - self.mean[-dim_xs:]) @ self.R[-dim_xs:, -dim_xs:].T
            Rs = self.R.diag()[-dim_xs:]
        neglogdet = -Rs.log().sum().item()
        neglogdets = torch.full((xs.shape[0],), neglogdet, device=xs.device)
        return us, neglogdets
    
    def grad_Q(self, us: Tensor, subset: str = "first") -> Tuple[Tensor, Tensor, Tensor]:
        self._check_subset(subset)
        num_us, dim_us = us.shape
        xs, neglogdets = self.Q(us, subset)
        if subset == "first":
            dxdus = self.L[:dim_us, None, :dim_us].repeat(1, num_us, 1)
        else:
            dxdus = self.L[-dim_us:, None, -dim_us:].repeat(1, num_us, 1)
        return xs, neglogdets, dxdus