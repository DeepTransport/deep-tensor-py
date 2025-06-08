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

        if reference is None:
            reference = GaussianReference()

        def Q(us: Tensor, subset: str) -> Tensor:
            return us
        
        def Q_inv(xs: Tensor, subset: str) -> Tensor:
            return xs
        
        def neglogdet_Q(us: Tensor, subset: str) -> Tensor:
            return torch.zeros(us.shape[0])
        
        def neglogdet_Q_inv(xs: Tensor, subset: str) -> Tensor: 
            return torch.zeros(xs.shape[0])
        
        Preconditioner.__init__(
            self, 
            reference=reference, 
            Q=Q,
            Q_inv=Q_inv,
            neglogdet_Q=neglogdet_Q,
            neglogdet_Q_inv=neglogdet_Q_inv,
            dim=dim
        )

        return