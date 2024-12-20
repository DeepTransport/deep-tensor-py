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


def compute_f_divergence(
    log_proposal: torch.Tensor, 
    log_target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes approximations of a set of f-divergences between two 
    probability distributions using samples.

    Parameters
    ----------
    log_proposal:
        An n-dimensional vector containing the proposal density (i.e., 
        the density the samples are drawn from) evaluated at each 
        sample.
    log_target:
        An m * n matrix. Each row should contain the values of a target
        density evaluated at each sample.

    Returns
    -------
    div_kl: 
        An approximation of the KL divergence between the distributions 
        based on the samples.
    div_h2:
        An approximation of the (squared) Hellinger distance between 
        the distributions based on the samples.
    div_tv:
        An approximation of the total variation distance between the 
        distributions based on the samples.
    
    TODO: derive the TV and H2 estimates.
        
    """

    log_target = torch.atleast_2d(log_target)
    m, n = log_target.shape
    log_proposal = log_proposal.tile((m, 1))

    log_ratios = log_target - log_proposal
    log_norm_ratios = compute_norm_const_ratio(log_ratios)

    div_kl = log_ratios.sum(dim=1) / n + log_norm_ratios
    div_h2 = 1.0 - (compute_norm_const_ratio(0.5*log_ratios) - 0.5*log_norm_ratios).exp()
    div_tv = 0.5 * (torch.exp(log_ratios + log_norm_ratios) - 1.0).abs().sum(dim=1) / n

    return div_kl, div_h2, div_tv


def compute_norm_const_ratio(
    log_ratios: torch.Tensor
) -> torch.Tensor:
    """Estimates the ratio of the normalising constants between two 
    (unnormalised) densities using a set of samples.
    
    Parameters
    ----------
    log_ratios:
        An m * n matrix. Each row contains the logarithm of the ratio 
        between a (possibly unnormalised) target density and the 
        proposal density, for samples drawn from the proposal density.
    
    Returns
    -------
    log_norm_ratios:
        An m-dimensional vector containing estimates of the log of the 
        ratio of the normalising constants between each of the target 
        densities and the proposal density.
    
    """

    m, n = log_ratios.shape
    log_norm_ratios = torch.zeros(m)

    for i in range(m):
        # Shift by maximum value to avoid numerical issues
        max_val = log_ratios[i].max()
        log_norm_ratios[i] = (max_val 
                              + (log_ratios[i] - max_val).exp().sum().log() 
                              - torch.tensor(n).log())
    
    return log_norm_ratios


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
    H: torch.Tensor, 
    tol: float=1e-2,
    max_iter: int=200
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a dominant r*r submatrix within an n*r matrix.
    
    Parameters
    ----------
    H:
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
    A:
        The product of the original matrix and the inverse of the 
        submatrix.

    References
    ----------
    Goreinov, SA, et al. (2010). How to find a good submatrix.

    """

    _, r = H.shape
    indices = lu_deim(H)[:r]

    if (rank := torch.linalg.matrix_rank(H[indices])) < r:
        msg = f"Initial submatrix is singular (rank {rank} < {r})."
        raise Exception(msg)

    A = torch.linalg.solve(H[indices].T, H.T).T

    for _ in range(max_iter):

        ij_max = A.abs().argmax(axis=None)
        i, j = torch.unravel_index(ij_max, A.shape)
        i_old = indices[j]

        if A[i, j].abs() < 1.0 + tol:
            # print(torch.max(A @ torch.linalg.inv(A[indices[:r]])))
            return indices, A

        A -= torch.outer(A[:, j], (A[i, :] - A[i_old, :]) / A[i, j])
        indices[j] = i

    msg = f"maxvol failed to converge in {max_iter} iterations."
    warnings.warn(msg)
    return indices, A