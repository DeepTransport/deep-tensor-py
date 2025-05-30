import torch 
from torch import Tensor

from .preconditioner import Preconditioner
from ..references import GaussianReference


class FixedLikelihoodPreconditioner(Preconditioner):
    r"""A preconditioner where the likelihood is fixed at a nominal value.

    some assumptions around gaussian noise, etc.
    
    TODO: finish docstring.

    could pass prior preconditioner into here. The full mapping and 
    its inverse are then trivial to compute...

    """

    def __init__(
        self, 
        y0: Tensor, 
        cov_y: Tensor,
        reference: GaussianReference | None = None
    ):
        
        raise NotImplementedError("not implemented yet...")
        return -1