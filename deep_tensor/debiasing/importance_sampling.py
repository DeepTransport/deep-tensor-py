import dataclasses 
import math

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
    particle $i$ [see, e.g., @Owen2013].

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

    n = log_weights.numel()
    log_weights = log_weights - log_weights.max()
    
    ess = log_weights.exp().sum().square() / (2.0*log_weights).exp().sum()
    ess_ratio = ess / n
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
        A structure containing the log-importance weights, the estimate 
        of the log-normalising constant of the target density (if 
        `self_normalised=False`), and the effective sample size.
    
    """
    log_weights = neglogfxs_irt - neglogfxs_exact
    n = log_weights.numel()
    
    if self_normalised:
        log_norm = torch.tensor(0.0)
    else:
        log_norm = log_weights.logsumexp(dim=0) - math.log(n)
        # log_weights = log_weights - log_norm

    ess = n * estimate_ess_ratio(log_weights)
    res = ImportanceSamplingResult(log_weights, log_norm, ess)
    return res