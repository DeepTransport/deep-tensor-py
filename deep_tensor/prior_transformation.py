from typing import Callable

from torch import Tensor

from .references import Reference


class PriorTransformation():
    """A mapping between the prior and a reference random variable.
    
    TODO: eventually, this will need to be able to evaluate the 
    action of the Jacobian of the Q^{-1}.

    TODO: perhaps this could be a dataclass.
    
    Parameters
    ----------
    reference:
        The density of the reference random variable.
    Q:
        A function which maps from the prior to the reference 
        distribution.
    Q_inv: 
        The inverse of `Q`.
    dim: 
        The dimension of the parameter.

    """

    def __init__(
        self, 
        reference: Reference,
        Q: Callable[[Tensor], Tensor],
        Q_inv: Callable[[Tensor], Tensor],
        neglogabsdet_Q_inv: Callable[[Tensor], Tensor],
        dim: int
    ):
        self.reference = reference
        self.Q = Q
        self.Q_inv = Q_inv
        self.neglogabsdet_Q_inv = neglogabsdet_Q_inv
        self.dim = dim
        return