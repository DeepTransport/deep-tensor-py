from typing import Tuple
import warnings

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


def deim(U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes a submatrix of a matrix of left singular vectors using
    the discrete empirical interpolation method (DEIM).
    
    Parameters
    ----------
    U: 
        An n * r matrix of left singular vectors, where n > r. The 
        vectors should be in order of decreasing singular value.
        
    Returns
    -------
    indices:
        The indices of the submatrix found using the DEIM.
    B: 
        The product of U and the inverse of the submatrix found using 
        the DEIM.
    
    References
    ----------
    Chaturantabut, S and Sorensen, DC (2010). Nonlinear model reduction 
    via discrete empirical interpolation.

    """

    n, r = U.shape 

    if r > n:
        msg = ("The number of rows of the input matrix "
               + "must be greater than or equal to the "
               + f"number of columns ({n} vs {r}).")
        raise Exception(msg)

    indices = torch.zeros(r, dtype=torch.int32)
    P = torch.zeros((n, r))

    indices[0] = U[:, 0].abs().argmax()
    P[indices[0], 0] = 1.0

    for i in range(1, r):

        P_i = P[:, :i]
        U_i = U[:, :i]

        c = torch.linalg.solve(P_i.T @ U_i, P_i.T @ U[:, i])
        r = U[:, i] - U[:, :i] @ c
        indices[i] = r.abs().argmax()
        P[indices[i], i] = 1.0

    B = torch.linalg.solve(U[indices].T, U.T).T
    return indices, B


def lu_deim(A):
    """TODO: tidy this up. Also test to see whether this is the same as 
    DEIM when applied with spectral polynomials."""

    n, r = A.shape

    if r > n:
        msg = ("The number of rows of the input matrix "
               + "must be greater than or equal to the "
               + f"number of columns ({n} vs {r}).")
        raise Exception(msg)

    indices = torch.arange(n)

    pivots = torch.linalg.lu_factor_ex(A)[1] - 1
    for i, p in enumerate(pivots):
        indices[[i, p.item()]] = indices[[p.item(), i]]

    return indices


def maxvol(
    A: torch.Tensor, 
    tol: float=1e-2,
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
    indices:
        The row indices of the dominant submatrix. 
    B:
        The product of the original matrix and the inverse of the 
        submatrix.

    References
    ----------
    Goreinov, SA, et al. (2010). How to find a good submatrix.

    """

    _, r = A.shape
    indices = lu_deim(A)[:r]

    if (rank := torch.linalg.matrix_rank(A[indices])) < r:
        msg = f"Initial submatrix is singular (rank {rank} < {r})."
        raise Exception(msg)

    B = torch.linalg.solve(A[indices].T, A.T).T

    for _ in range(max_iter):

        ij_max = torch.argmax(torch.abs(B), axis=None)
        i, j = torch.unravel_index(ij_max, B.shape)
        i_old = indices[j]

        if torch.abs(B[i, j]) < 1.0 + tol:
            # print(torch.max(A @ torch.linalg.inv(A[indices[:r]])))
            return indices, B

        B -= torch.outer(B[:, j], (B[i, :] - B[i_old, :]) / B[i, j])
        indices[j] = i

    msg = f"maxvol failed to converge in {max_iter} iterations."
    warnings.warn(msg)
    return indices, B