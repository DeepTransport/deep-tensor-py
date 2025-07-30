from typing import Sequence, Tuple
import warnings

import torch
from torch import Tensor 


def batch_mul(*arrs: Tensor) -> Tensor:
    """Batch-multiplies a list of three-dimensional tensors together."""

    for a in arrs:
        if a.ndim != 3:
            msg = "All input tensors must be three-dimensional."
            raise Exception(msg)
        
    if len(arrs) == 1:
        return arrs[0]

    prod = arrs[0]
    for a in arrs[1:]:
        prod = torch.einsum("...ij, ...jk", prod, a)
    return prod


def cartesian_prod(arrs: Sequence[Tensor]) -> Tensor:
    """Computes the Cartesian product associated with a set of 2D 
    tensors.

    Parameters
    ----------
    arrs:
        A list of two-dimensional tensors. Any tensors with no elements 
        will be filtered out.

    Returns
    -------
    prod:
        A two-dimensional tensor containing the Cartesian product of 
        the tensors.
    
    """

    # Ignore tensors with no elements
    arrs = [a for a in arrs if a.numel() > 0]
    arrs = [torch.atleast_2d(a) for a in arrs]

    if not arrs:
        msg = "List of empty tensors found."
        warnings.warn(msg)
        return torch.tensor([[]])
    
    if len(arrs) == 1:
        return arrs[0]

    prod = arrs[0]
    for arr in arrs[1:]:
        prod = torch.tensor([[*p, *a] for p in prod for a in arr])

    return prod


def unfold_left(H: Tensor) -> Tensor:
    """Forms the left unfolding matrix of a three-dimensional tensor.
    """
    if H.ndim != 3:
        msg = "Input tensor must be 3-dimensional."
        raise Exception(msg)
    r_p, n_k, r_k = H.shape
    H = H.reshape(r_p * n_k, r_k)
    return H


def unfold_right(H: Tensor) -> Tensor:
    """Forms the transpose of the right unfolding matrix of a 
    3-dimensional tensor.
    """
    if H.ndim != 3:
        msg = "Input tensor must be 3-dimensional."
        raise Exception(msg)
    r_p, n_k, r_k = H.shape
    H = H.swapdims(0, 2).reshape(n_k * r_k, r_p)
    return H


def fold_left(H: Tensor, newshape: Sequence) -> Tensor:
    """Computes the inverse of the unfold_left operation.
    """
    if H.ndim > 2:
        msg = "Dimension of input tensor cannot be greater than 2."
        raise Exception(msg)
    H = H.reshape(*newshape)
    return H


def fold_right(H: Tensor, newshape: Sequence) -> Tensor:
    """Computes the inverse of the unfold_right operation.
    """
    if H.ndim > 2:
        msg = "Dimension of input tensor cannot be greater than 2."
        raise Exception(msg)
    H = H.reshape(*reversed(newshape)).swapdims(0, 2)
    return H


def tsvd(
    H: Tensor, 
    tol: float = 0.0, 
    max_rank: int | None = None
) -> Tuple[Tensor, Tensor, Tensor, int]:
    """Computes the truncated SVD of a matrix.
    
    Parameters
    ----------
    H:
        An m * n matrix to compute the truncated SVD of.
    tol:
        The tolerance used when truncating the singular values. The 
        minimum number of singular values such that the sum of their 
        squares exceeds (1 - tol) will be retained.
    max_rank:
        An optional hard upper limit on the number of singular values, 
        r, to retain.
    
    Returns
    -------
    Ur: 
        An m * r matrix containing the retained left singular vectors.
    sr:
        An r-dimensional vector containing the retained singular values.
    Vhr: 
        An n * r matrix containing the transpose of the retained right 
        singular vectors.
    
    """

    U, s, Vh = torch.linalg.svd(H, full_matrices=False)
            
    energies = torch.cumsum(s**2, dim=0)
    max_energy = energies[-1].clone()
    energies /= max_energy

    rank = int((torch.sum(energies < 1.0 - tol) + 1).clamp(max=s.numel()))

    if max_rank is not None:
        rank = min(rank, max_rank)
    
    return U[:, :rank], s[:rank], Vh[:rank, :], rank