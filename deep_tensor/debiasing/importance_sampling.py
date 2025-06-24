import dataclasses 

import torch
from torch import Tensor


@dataclasses.dataclass
class ImportanceSamplingResult(object):
    r"""An object containing the results of importance sampling.
    
    Attributes
    ----------
    log_weights: Tensor
        An $n$-dimensional vector containing the unnormalised 
        importance weights associated with a set of samples.
    log_norm: Tensor
        An estimate of the logarithm of the normalising constant 
        associated with the target density.
    ess: Tensor
        An estimate of the effective sample size. 

    Notes
    -----
    The effective sample size is computed using the formula
    $$
        N_{\mathrm{eff}} = \frac{(\sum_{i=1}^{n}w_{i})^{2}}{\sum_{i=1}^{n}w_{i}^{2}},
    $$
    where $w_{i}$ denotes the importance weight associated with 
    particle $i$ (Owen, 2013).

    References
    ----------
    Owen, AB (2013, Chapter 6). *[Monte Carlo theory, methods and 
    examples](https://artowen.su.domains/mc/)*.

    """
    log_weights: Tensor
    log_norm: Tensor 
    ess: Tensor


def estimate_ess_ratio(log_weights: Tensor) -> Tensor:
    """Returns the ratio of the effective sample size to the number of
    particles.

    Parameters
    ----------
    log_weights:
        A vector containing the logarithm of the ratio between the 
        target density and the proposal density evaluated for each 
        sample. 

    Returns
    -------
    ess_ratio:
        The ratio of the effective sample size to the number of 
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
        log_norm = torch.tensor(0.0)
    else: 
        # Estimate normalising constant of the target density, then 
        # shift the log-weights (for better numerics) before normalising
        log_norm = log_weights.exp().mean().log()
        log_weights = log_weights - log_weights.max()
        log_weights = log_weights - log_weights.exp().mean().log()

    ess = log_weights.numel() * estimate_ess_ratio(log_weights)
    res = ImportanceSamplingResult(log_weights, log_norm, ess)
    return res