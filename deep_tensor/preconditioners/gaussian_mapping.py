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

    def Q(self, us: Tensor, subset: str = "first") -> Tensor:
        self._check_subset(subset)
        d_us = us.shape[1]
        if subset == "first":
            xs = self.mean[:d_us] + (us @ self.L[:d_us, :d_us].T)
        else:
            xs = self.mean[-d_us:] + (us @ self.L[-d_us:, -d_us:].T)
        return xs
    
    def Q_inv(self, xs: Tensor, subset: str = "first") -> Tensor:
        self._check_subset(subset)
        d_xs = xs.shape[1]
        if subset == "first":
            us = (xs - self.mean[:d_xs]) @ self.R[:d_xs, :d_xs].T
        else:
            xs = (xs - self.mean[-d_xs:]) @ self.R[-d_xs:, -d_xs:].T
        return us
    
    def neglogdet_Q(self, us: Tensor, subset: str = "first") -> Tensor:
        self._check_subset(subset)
        d_us = us.shape[1]
        if subset == "first":
            Ls = self.L.diag()[:d_us]
        else:
            Ls = self.L.diag()[-d_us:]
        neglogdets = torch.full((us.shape[0],), -Ls.log().sum().item())
        return neglogdets 
    
    def neglogdet_Q_inv(self, xs: Tensor, subset: str = "first") -> Tensor: 
        self._check_subset(subset)
        d_xs = xs.shape[1]
        if subset == "first":
            Rs = self.R.diag()[:d_xs]
        else:
            Rs = self.R.diag()[-d_xs:]
        neglogdets = torch.full((xs.shape[0],), -Rs.log().sum().item())
        return neglogdets 