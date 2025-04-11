from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from .references import Reference


@dataclass
class PriorTransformation():
    r"""A mapping between the prior and a reference random variable.

    
    
    Parameters
    ----------
    reference:
        The density of the reference random variable.
    Q:
        A function which maps from the prior to the reference 
        distribution.
    Q_inv: 
        The inverse of `Q`.
    neglogabsdet_Q_inv:
        A function which takes as input an $n \times d$ matrix
    dim: 
        The dimension of the parameter.

    """

    reference: Reference
    Q: Callable[[Tensor], Tensor]
    Q_inv: Callable[[Tensor], Tensor]
    neglogabsdet_Q_inv: Callable[[Tensor], Tensor]
    dim: int