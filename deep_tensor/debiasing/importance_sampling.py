from dataclasses import dataclass

from torch import Tensor

from ..tools import compute_ess_ratio


@dataclass
class ImportanceSamplingResult(object):
    log_weights: Tensor
    log_norm: Tensor 
    ess: Tensor


def run_importance_sampling(
    neglogfxs_irt: Tensor,
    neglogfxs_exact: Tensor,
    self_normalised: bool = False
) -> ImportanceSamplingResult:
    r"""Computes the importance weights associated with a set of samples.

    Parameters
    ----------
    neglogfxs_irt:
        An $n$-dimensional vector containing the potential function 
        associated with the DIRT object evaluated at each sample.
    neglogfxs_exact:
        An $n$-dimensional vector containing the potential function 
        associated with the target density evaluated at each sample.
    self_normalised:
        Whether the target density is normalised. If not, the log of 
        the normalising constant will be estimated using the weights. 

    Returns
    -------
    res: 
        A structure containing the log-importance weights (normalised, 
        if `self_normalised=False`), the estimate of the 
        log-normalising constant of the target density (if 
        `self_normalised=False`), and the effective sample size.
    
    """
    log_weights = neglogfxs_irt - neglogfxs_exact
    
    if self_normalised:
        log_norm = 0.0
    else: 
        # Estimate normalising constant of the target density, then 
        # shift the log-weights (for better numerics) before normalising
        log_norm = log_weights.exp().sum().log()
        log_weights = log_weights - log_weights.max()
        log_weights = log_weights - log_weights.exp().sum().log()

    ess = log_weights.numel() * compute_ess_ratio(log_weights)
    res = ImportanceSamplingResult(log_weights, log_norm, ess)
    return res