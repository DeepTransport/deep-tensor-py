import torch
from torch import Tensor

from .preconditioner import Preconditioner
from .gaussian_preconditioner import GaussianPreconditioner
from ..references import GaussianReference


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
    perturb_eigvals:
        If this is set to True, the diagonal of the sample covariance 
        matrix will be multiplied by a number slightly greater than 1. 
        This will ensure it is positive definite.
    inflation:
        TODO: write this.

    """

    def __init__(
        self, 
        samples: Tensor, 
        reference: GaussianReference | None = None,
        perturb_eigvals: bool = False,
        inflation: Tensor | None = None
    ):
        
        mean = torch.mean(samples, axis=0)
        cov = torch.cov(samples.T)

        if perturb_eigvals:
            cov += 1e-8 * cov.diag().diag()

        if inflation is not None:
            if not isinstance(inflation, Tensor):
                inflation = torch.tensor(inflation)
            if inflation.numel() == 1:
                cov = inflation.square() * cov
            else:
                inflation = torch.diag(inflation)
                cov = inflation @ cov @ inflation

        GaussianPreconditioner.__init__(self, mean, cov, reference)
        return