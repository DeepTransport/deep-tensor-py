from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from ..references import Reference


@dataclass
class Preconditioner():
    r"""A user-defined preconditioning function.
    Ideally, the pushforward of the reference density under the 
    preconditioning function will be as similar as possible to the 
    target density; this makes the subsequent construction of the DIRT 
    approximation to the target density more efficient.

    The mapping, which we denote using $Q(\cdot)$, needs to be 
    invertible. There are additional benefits if the mapping is lower 
    or upper triangular (or both):

      - If the mapping is lower triangular, one can evaluate the 
        marginal densities of the corresponding DIRT object in the 
        first $k$ variables, and condition on the first $k$ variables, 
        where $1 \leq k < d$.
      - If the mapping is upper triangular, one can evaluate the 
        marginal densities of the corresponding DIRT object in the 
        last $k$ variables, and condition on the final $k$ variables, 
        where $1 \leq k < d$.

    Parameters
    ----------
    reference:
        The density of the reference random variable.
    Q:
        A function which takes an $n \times k$ matrix containing 
        samples from the reference domain and a string indicating 
        whether these are samples of the first (`subset='first'`) or 
        last (`subset='last'`) $k$ variables, and returns an 
        $n \times k$ matrix containing samples from the approximation 
        domain, after applying the mapping $Q(\cdot)$ to each sample.
    Q_inv: 
        A function which takes an $n \times k$ matrix containing 
        samples from the approximation domain and a string indicating 
        whether these are samples of the first (`subset='first'`) or 
        last (`subset='last'`) $k$ variables, and returns an 
        $n \times k$ matrix containing samples from the reference 
        domain, after applying the mapping $Q^{-1}(\cdot)$ to each 
        sample.
    neglogdet_Q:
        A function which takes an $n \times k$ matrix containing 
        samples from the reference domain and a string indicating 
        whether these are samples of the first (`subset='first'`) or 
        last (`subset='last'`) $k$ variables, and returns an 
        $n$-dimensional vector containing the negative log-determinant 
        of $Q(\cdot)$ evaluated at each sample.
    neglogdet_Q_inv:
        A function which takes an $n \times k$ matrix containing 
        samples from the approximation domain and a string indicating 
        whether these are samples of the first (`subset='first'`) or 
        last (`subset='last'`) $k$ variables, and returns an 
        $n$-dimensional vector containing the negative log-determinant 
        of $Q^{-1}(\cdot)$ evaluated at each sample.
    dim: 
        The dimension, $d$, of the target (and reference) random 
        variable.

    """
    
    reference: Reference
    Q: Callable[[Tensor, str], Tensor]
    Q_inv: Callable[[Tensor, str], Tensor]
    neglogdet_Q: Callable[[Tensor, str], Tensor]
    neglogdet_Q_inv: Callable[[Tensor, str], Tensor]
    dim: int