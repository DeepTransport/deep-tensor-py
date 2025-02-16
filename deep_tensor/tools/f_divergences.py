from typing import Tuple 

import torch
from torch import Tensor


def compute_norm_const_ratio(log_ratios: Tensor) -> Tensor:
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
        log_norm_ratios[i] = ((log_ratios[i] - max_val).exp().sum().log() 
                              + max_val
                              - torch.tensor(n).log())
    
    return log_norm_ratios


def compute_f_divergence(
    log_proposal: Tensor, 
    log_target: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
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