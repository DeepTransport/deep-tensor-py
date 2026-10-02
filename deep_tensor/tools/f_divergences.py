import math
import warnings

import torch
from torch import Tensor


def estimate_dhell(
    neglogfxs: Tensor, 
    negloggxs: Tensor,
    negloghxs: Tensor | None = None
) -> Tensor:
    """Estimates the Hellinger divergence between two (unnormalised)
    probability densities using an importance sampling estimate.

    Parameters
    ----------
    neglogfxs:
        An n-dimensional vector containing evaluations of the negative 
        logarithm of the first (possibly unnormalised) density.
    negloggxs:
        An n-dimensional vector containing evaluations of the negative 
        logarithm of the second (possibly unnormalised) density.
    negloghs:
        An n-dimensional vector containing evaluations of the negative 
        logarithm of the proposal density. If this is not supplied, 
        neglogfxs will be assumed to be draws from the (normalised) 
        importance density.
    
    Returns
    -------
    dhell:
        An importance sampling estimate of the Hellinger divergence.
    
    """

    if negloghxs is None:
        negloghxs = neglogfxs.clone()

    n = neglogfxs.numel()
    neglogfx_norm = -torch.logsumexp(negloghxs - neglogfxs, dim=0) + math.log(n)
    negloggx_norm = -torch.logsumexp(negloghxs - negloggxs, dim=0) + math.log(n)

    neglogfxs_norm = neglogfxs - neglogfx_norm
    negloggxs_norm = negloggxs - negloggx_norm

    dhell_sq = 1.0 - torch.exp(
        torch.logsumexp(-0.5*neglogfxs_norm-0.5*negloggxs_norm+negloghxs, dim=0) 
        - math.log(n)
    )
    dhell = dhell_sq.clamp(min=0.0) ** 0.5
    return dhell


DIVERGENCES = ("h2", "kl", "tv")


def compute_log_norm(log_ratios: Tensor) -> Tensor:
    """Estimates the normalising constant of a given target density.
    
    Parameters
    ----------
    log_ratios:
        An n-dimensional vector, containing the logarithm of the ratio 
        between the (unnormalised) target density and the (normalised)
        proposal density, for samples drawn from the proposal density.
    
    Returns
    -------
    log_norm_ratio:
        The estimate of the log of the normalising constant of the 
        target density.
    
    """
    # Shift by maximum value to avoid numerical issues
    max_val = log_ratios.max()
    log_norm_ratio = (log_ratios - max_val).exp().mean().log() + max_val
    return log_norm_ratio


def compute_f_divergence(logqs: Tensor, logps: Tensor, div: str = "h2") -> Tensor:
    """Computes approximations of a set of f-divergences between two 
    probability densities using samples.

    Parameters
    ----------
    logqs:
        An n-dimensional vector containing the (normalised) proposal 
        density (i.e., the density the samples are drawn from) 
        evaluated at each sample.
    logps:
        An n-dimensional vector containing the values of the other 
        (unnormalised) density evaluated at each sample.
    div:
        The type of divergence to estimate. Can be 'h2' (squared 
        Hellinger distance), 'kl' (reversed KL divergence) or 'tv' 
        (total variation distance).

    Returns
    -------
    f_div: 
        The estimate of the requested f-divergence using the provided 
        evaluations of the densities.
    
    References
    ----------
    https://en.wikipedia.org/wiki/F-divergence#Common_examples_of_f-divergences
        
    """
    
    msg = "This function is deprecated. Please use `estimate_dhell` instead."
    warnings.warn(msg)

    div = div.lower()
    if div not in DIVERGENCES:
        msg = (
            f"Divergence '{div}' not recognised. Recognised values are "
            ", ".join(DIVERGENCES) + "."
        )
        raise Exception(msg)

    log_ratios = logps - logqs
    log_norm = compute_log_norm(log_ratios)

    if div == "h2":
        f_div = 1.0 - (compute_log_norm(0.5*log_ratios) - 0.5*log_norm).exp()
    elif div == "kl":
        f_div = -log_ratios.mean() + log_norm
    elif div == "tv": 
        f_div = 0.5 * (torch.exp(log_ratios - log_norm) - 1.0).abs().mean()
    
    f_div = torch.clamp(f_div, min=0.0)
    return f_div