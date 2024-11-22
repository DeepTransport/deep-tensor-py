from typing import Tuple

import torch


def reshape_matlab(A: torch.Tensor, newshape: Tuple) -> torch.Tensor:
    """https://stackoverflow.com/questions/63960352/reshaping-order-in-pytorch-fortran-like-index-ordering"""
    
    A = (A.permute(*reversed(range(A.ndim)))
          .reshape(*reversed(newshape))
          .permute(*reversed(range(len(newshape)))))
    
    return A


def compute_ess_ratio(log_weights: torch.Tensor) -> torch.Tensor:
    """Returns the ratio of the effective sample size to the number of
    particles.

    References
    ----------
    Owen, AB (2013). Monte Carlo theory, methods and examples. Chapter 9.

    """

    sample_size = log_weights.numel()
    log_weights = log_weights - log_weights.max()

    ess = log_weights.exp().sum().square() / (2.0*log_weights).exp().sum()
    ess_ratio = ess / sample_size
    return ess_ratio


def maxvol(
    A: torch.Tensor, 
    tol: float=1e-6,
    max_iter: int=200
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a dominant r*r submatrix within an n*r matrix.
    
    Parameters
    ----------
    A:
        n*r matrix, where n > r.
    tol:
        Convergence tolerance. The algorithm is considered converged if
        the absolute value of the largest element in A^{-1} @ B (where 
        B is the submatrix identified) is no greater than 1 + tol.
    max_iter:
        The maximum number of iterations to carry out.

    Returns
    -------
    :
        The row indices of the dominant submatrix, and the product of 
        the original matrix and the inverse of the submatrix.

    References
    ----------
    Goreinov, SA, et al. (2010). How to find a good submatrix.

    """

    # TODO: implement initial guess using QDEIM.

    n, r = A.shape
    indices = torch.arange(n)

    pivots = torch.linalg.lu_factor_ex(A)[1] - 1
    for i, p in enumerate(pivots):  # Swap rows (why -1)?
        indices[i], indices[p] = indices[p].clone(), indices[i].clone()

    if r > n:
        msg = ("The number of rows of the input matrix "
               + "must be greater than or equal to the "
               + f"number of columns ({n} vs {r}).")
        raise Exception(msg)

    if (rank := torch.linalg.matrix_rank(A[indices[:r]])) < r:
        msg = f"Initial submatrix is singular (rank {rank} < {r})."
        raise Exception(msg)

    B = A @ torch.linalg.inv(A[indices[:r]])
    I = torch.eye(n)

    # indices = torch.arange(n)

    for _ in range(max_iter):

        # Find entry of B with greatest absolute value
        ij_max = torch.argmax(torch.abs(B), axis=None)
        i_max, j_max = torch.unravel_index(ij_max, B.shape)
        b_ij = B[i_max, j_max]

        # Check for convergence
        if torch.abs(b_ij) < 1.0 + tol:
            # print(torch.max(A @ torch.linalg.inv(A[indices[:r]])))
            return indices[:r], B
        
        # Update indices of maxvol matrix
        indices[i_max], indices[j_max] = indices[j_max].clone(), indices[i_max].clone()

        # # Make rank 1 update to B matrix
        # B1 = (B[r:][:, j_max] + I[r:][:, i_max])[:, None]
        # B2 = (B[i_max, :] - I[j_max, :r])[None, :]

        # B[r:] -= B1 @ B2 / b_ij
        B = A @ torch.linalg.inv(A[indices[:r]])

    msg = f"maxvol failed to converge in {max_iter} iterations."
    # warnings.warn(msg)
    return indices[:r], B