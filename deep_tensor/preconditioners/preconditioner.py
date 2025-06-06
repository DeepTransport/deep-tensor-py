from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from ..references import Reference


@dataclass
class Preconditioner():
    r"""A mapping between the prior and a reference random variable.

    This mapping needs to be triangular. 
      - If the mapping is lower triangular, one can evaluate the 
        marginal densities of the corresponding DIRT object in the 
        first $k$ variables, and condition on the first $k$ variables, 
        where $1 \leq k \leq d$.
      - If the mapping is upper triangular, one can evaluate the 
        marginal densities of the corresponding DIRT object in the 
        last $k$ variables, and condition on the final $k$ variables, 
        where $1 \leq k \leq d$.
    
    Parameters
    ----------
    reference:
        The density of the reference random variable.
    Q:
        A function which maps from the reference distribution to the 
        prior.
    Q_inv: 
        The inverse of `Q`.
    neglogdet_Q:
        TODO
    neglogdet_Q_inv:
        TODO
    dim: 
        The dimension of the parameter.

    """

    reference: Reference
    Q: Callable[[Tensor], Tensor]
    Q_inv: Callable[[Tensor], Tensor]
    neglogdet_Q: Callable[[Tensor], Tensor]
    neglogdet_Q_inv: Callable[[Tensor], Tensor]
    dim: int