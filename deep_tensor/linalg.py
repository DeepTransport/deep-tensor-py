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

    prod = arrs[0]
    for a in arrs[1:]:
        prod = torch.einsum("...ij, ...jk", prod, a)
    return prod


def cartesian_prod(*arrs: Tensor) -> Tensor:
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
    matrices = [a for a in arrs if a.numel() > 0]
    matrices = [torch.atleast_2d(m) for m in matrices]

    if not matrices:
        msg = "List of empty tensors found."
        warnings.warn(msg)
        return torch.tensor([[]])
    
    if len(matrices) == 1:
        return matrices[0]

    prod = matrices[0]
    for matrix in matrices[1:]:
        prod = torch.tensor([[*row_p, *row_m] 
                             for row_p in prod 
                             for row_m in matrix])
    return prod


def n_mode_prod(T: Tensor, M: Tensor, n: int) -> Tensor:
    """Computes the n-mode product of a tensor and a matrix or vector, 
    as in Kolda and Bader (2009).
    """

    if M.ndim not in (1, 2):
        raise Exception("M must be a matrix or vector.")

    prod = T.swapdims(n, -1)

    if M.ndim == 1:
        prod = torch.einsum("...ij,j", prod, M)
    else:
        prod = torch.einsum("...ij, kj", prod, M)
        prod = prod.swapdims(n, -1)
    
    return prod


def mode_n_unfolding(T: Tensor, n: int) -> Tensor:
    """Computes the mode-n unfolding of a tensor. Each row of the 
    returned matrix contains a mode-n fibre of the input tensor.
    """
    num_fibres = T.shape[n]
    T = T.swapdims(n, -1)
    T = T.reshape(-1, num_fibres)
    return T


def mode_n_folding(M: Tensor, n: int, newshape: Sequence) -> Tensor:
    """Computes the inverse of the mode-n unfolding operation."""

    newshape = list(newshape)
    newshape[n], newshape[-1] = newshape[-1], newshape[n]

    T = M.reshape(*newshape)
    T = T.swapdims(n, -1)
    return T


def unfold_left(T: Tensor) -> Tensor:
    """Forms the left unfolding matrix of a three-dimensional tensor.
    """
    if T.ndim != 3:
        msg = "Input tensor must be 3-dimensional."
        raise ValueError(msg)
    r_p, n_k, r_k = T.shape
    T = T.reshape(r_p * n_k, r_k)
    return T


def unfold_right(T: Tensor) -> Tensor:
    """Forms the transpose of the right unfolding matrix of a 
    3-dimensional tensor.
    """
    if T.ndim != 3:
        msg = "Input tensor must be 3-dimensional."
        raise ValueError(msg)
    r_p, n_k, r_k = T.shape
    T = T.swapdims(0, 2).reshape(n_k * r_k, r_p)
    return T


def fold_left(H: Tensor, newshape: Sequence) -> Tensor:
    """Computes the inverse of the unfold_left operation.
    """
    if H.ndim > 2:
        msg = "Dimension of input tensor cannot be greater than 2."
        raise ValueError(msg)
    H = H.reshape(*newshape)
    return H


def fold_right(H: Tensor, newshape: Sequence) -> Tensor:
    """Computes the inverse of the unfold_right operation.
    """
    if H.ndim > 2:
        msg = "Dimension of input tensor cannot be greater than 2."
        raise ValueError(msg)
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

    if tol == 0.0:
        rank = s.numel()
    else:
        rank = torch.sum(energies <= 1.0 - tol) + 1
        rank = int(rank.clamp(max=s.numel()))

    if max_rank is not None:
        rank = min(rank, max_rank)
    
    return U[:, :rank], s[:rank], Vh[:rank, :], rank