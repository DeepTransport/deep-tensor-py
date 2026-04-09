from torch import Tensor

from .gaussian_mapping import GaussianMapping
from ..references import GaussianReference


class SampleBasedMapping(GaussianMapping):
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

    """

    def __init__(
        self, 
        samples: Tensor, 
        reference: GaussianReference | None = None,
        perturb_eigvals: bool = False
    ):
        mean = samples.mean(dim=0)
        cov = samples.T.cov()
        if perturb_eigvals:
            cov += 1e-8 * cov.diag().diag()
        GaussianMapping.__init__(self, mean, cov, reference)
        return