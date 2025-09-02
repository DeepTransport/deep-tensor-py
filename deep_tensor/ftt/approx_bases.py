from typing import List, Tuple

import torch
from torch import Tensor

from ..polynomials import Basis1D


class ApproxBases():
    """An object that contains a polynomial basis in each dimension, 
    used to construct a FTT approximation.

    Parameters
    ----------
    bases:
        A single set of basis functions (to be used in all dimensions),
        or a list of basis functions to be used in each dimension.
    dim:
        The dimension of the function being approximated.
    
    """

    def __init__(self, bases: Basis1D | List[Basis1D], dim: int):
        
        if isinstance(bases, Basis1D):
            bases = [bases]
        if len(bases) == 1:
            bases *= dim
        if len(bases) != dim:
            msg = (
                "Number of polynomial bases passed in does not equal "
                f"specified dimension (expected {dim}, got {len(bases)})."
            )
            raise Exception(msg)

        self.bases = bases
        self.dim = dim
        return
    
    def __getitem__(self, k: int) -> Basis1D:
        return self.bases[k]
    
    @staticmethod
    def _check_indices_shape(inds: Tensor, xs: Tensor) -> None:
        """Confirms whether the length of a vector of indices is equal 
        to the dimension of a set of samples.
        """
        if inds.numel() != xs.shape[1]:
            msg = "Samples do not have the correct dimensions."
            raise Exception(msg)
        return

    def sample_measure(self, n: int) -> Tuple[Tensor, Tensor]:
        """Generates a set of random variates from the local weighting 
        function.
        """ 
        ls = torch.zeros((n, self.dim))
        neglogwls = torch.zeros(n)
        for k in range(self.dim):
            ls[:, k] = self.bases[k].sample_measure(n)
            neglogwls -= self.bases[k].eval_log_measure(ls[:, k])
        return ls, neglogwls

    def eval_measure_potential(
        self, 
        ls: Tensor, 
        inds: Tensor | None = None
    ) -> Tensor:
        """Computes the negative logarithm of the weighting function 
        associated with (a subset of) the basis functions (defined in 
        the local domain).
        """
        
        if inds is None:
            inds = torch.arange(self.dim)
        ApproxBases._check_indices_shape(inds, ls)
        
        neglogwls = torch.empty_like(ls)
        for i, ls_i in enumerate(ls.T):
            basis = self.bases[inds[i]]
            neglogwls[:, i] = -basis.eval_log_measure(ls_i)
        
        return neglogwls.sum(dim=1)

    def eval_measure_potential_grad(
        self, 
        ls: Tensor,
        inds: Tensor | None = None
    ) -> Tensor:
        """Computes the gradient of the negative logarithm of the 
        weighting functions of (a subset of) the basis functions for a 
        given set of samples in the local domain.
        """

        if inds is None:
            inds = torch.arange(self.dim)
        ApproxBases._check_indices_shape(inds, ls)
        
        negloggradwls = torch.empty_like(ls)
        for i, ls_i in enumerate(ls.T):
            basis = self.bases[inds[i]]
            negloggradwls[:, i] = -basis.eval_log_measure_deriv(ls_i)
        
        return negloggradwls