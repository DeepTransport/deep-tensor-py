import torch
from torch import Tensor
from torch import linalg

from .preconditioner import Preconditioner
from ..references import GaussianReference


class GaussianPreconditioner(Preconditioner):
    r"""Gaussian preconditioner..."""

    def __init__(
        self,
        mean: Tensor,
        cov: Tensor, 
        reference: GaussianReference | None = None, 
        perturb_eigvals: bool = False
    ):

        if reference is None:
            reference = GaussianReference()
        elif not isinstance(reference, GaussianReference):
            msg = "Reference density should be Gaussian."
            raise Exception(msg)

        self.mean = mean
        self.cov = cov 
        if perturb_eigvals:
            self.cov += 1e-8 * self.cov.diag().diag()
        self.L: Tensor = linalg.cholesky(self.cov)
        self.R: Tensor = linalg.inv(self.L)

        def Q(xs: Tensor) -> Tensor:
            d_xs = xs.shape[1]
            ms = self.mean[:d_xs] + (xs @ self.L[:d_xs, :d_xs].T)
            return ms 
        
        def Q_inv(ms: Tensor) -> Tensor:
            d_ms = ms.shape[1]
            xs = (ms - self.mean[:d_ms]) @ self.R[:d_ms, :d_ms].T
            return xs 
        
        def neglogdet_Q(xs: Tensor) -> Tensor:
            d_xs = xs.shape[1]
            neglogdets = -self.L.diag()[:d_xs].log().sum()
            return neglogdets 
        
        def neglogdet_Q_inv(ms: Tensor) -> Tensor: 
            d_ms = ms.shape[1]
            neglogdets = -self.R.diag()[:d_ms].log().sum()
            return neglogdets 
        
        Preconditioner.__init__(
            self, 
            reference=reference, 
            Q=Q,
            Q_inv=Q_inv,
            neglogdet_Q=neglogdet_Q,
            neglogdet_Q_inv=neglogdet_Q_inv,
            dim=mean.numel()
        )

        return

class SampleBasedPreconditioner(Preconditioner):
    r"""An approximate linear coupling between the reference and target densities.

    Builds an approximate linear coupling between the unit Gaussian 
    density and the joint density of the parameters and observations, 
    using a set of samples. 

    Parameters
    ----------
    samples:
        An $n \times d$ matrix containing a set of samples from the 
        target density.
    reference:
        The reference density. This must be a Gaussian density.

    """

    def __init__(
        self, 
        samples: Tensor, 
        reference: GaussianReference | None = None,
        perturb_eigvals: bool = False
    ):
        
        mean = torch.mean(samples, axis=0)
        cov = torch.cov(samples.T)
        ## TEMP
        # cov *= 4.0
        # inflations = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        # inflations = torch.diag(inflations)
        # C = inflations @ C @ inflations
        ## TEMP
        GaussianPreconditioner.__init__(
            self, 
            mean, 
            cov, 
            reference, 
            perturb_eigvals
        )

        return