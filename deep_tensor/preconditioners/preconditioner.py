import abc

from torch import Tensor

from ..references import Reference


class Preconditioner(abc.ABC):
    r"""The base class for preconditioning mappings.

    Ideally, the pushforward of the reference density under the 
    preconditioning mapping will be as similar as possible to the 
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
    dim: 
        The dimension, $d$, of the target (and reference) random 
        variable.

    Notes
    -----
    To construct a custom preconditioning mapping, the user must 
    construct a class derived from this class with methods `Q()`, 
    `Q_inv()`, `neglogdet_Q()`, and `neglogdet_Q_inv()`.

    """

    def __init__(self, reference: Reference, dim: int):
        self.reference = reference
        self.dim = dim 
        return
    
    @abc.abstractmethod
    def Q(self, us: Tensor, subset: str = "first") -> Tensor:
        r"""Applies the mapping $Q(\cdot)$ to a set of samples.

        Parameters
        ----------
        us:
            An $n \times k$ matrix containing samples from the 
            reference domain.
        subset:    
            If $k < d$, whether the samples are samples of the first 
            (`subset='first'`) or last (`subset='last'`) $k$ variables. 
            
        Returns
        -------
        xs:
            An $n \times k$ matrix containing samples from the 
            approximation domain, after applying the mapping $Q(\cdot)$ 
            to each sample.
        
        """
        pass

    @abc.abstractmethod
    def Q_inv(self, xs: Tensor, subset: str = "first") -> Tensor:
        r"""Applies the mapping $Q^{-1}(\cdot)$ to a set of samples.

        Parameters
        ----------
        xs:
            An $n \times k$ matrix containing samples from the 
            approximation domain.
        subset:    
            If $k < d$, whether the samples are samples of the first 
            (`subset='first'`) or last (`subset='last'`) $k$ variables. 
            
        Returns
        -------
        us:
            An $n \times k$ matrix containing samples from the 
            reference domain, after applying the mapping $Q^{-1}(\cdot)$ 
            to each sample.
        
        """
        pass

    @abc.abstractmethod
    def neglogdet_Q(self, us: Tensor, subset: str = "first") -> Tensor:
        r"""Applies the mapping $Q(\cdot)$ to a set of samples.

        Parameters
        ----------
        us:
            An $n \times k$ matrix containing samples from the 
            reference domain.
        subset:    
            If $k < d$, whether the samples are samples of the first 
            (`subset='first'`) or last (`subset='last'`) $k$ variables. 
            
        Returns
        -------
        neglogdets:
            An $n$-dimensional vector containing the negative 
            log-determinant of $Q(\cdot)$ evaluated at each sample.
        
        """
        pass

    @abc.abstractmethod
    def neglogdet_Q_inv(self, xs: Tensor, subset: str = "first") -> Tensor:
        r"""Applies the mapping $Q^{-1}(\cdot)$ to a set of samples.

        Parameters
        ----------
        xs:
            An $n \times k$ matrix containing samples from the 
            approximation domain.
        subset:    
            If $k < d$, whether the samples are samples of the first 
            (`subset='first'`) or last (`subset='last'`) $k$ variables. 
            
        Returns
        -------
        neglogdets:
            An $n$-dimensional vector containing the negative 
            log-determinant of $Q^{-1}(\cdot)$ evaluated at each sample.
        
        """
        pass