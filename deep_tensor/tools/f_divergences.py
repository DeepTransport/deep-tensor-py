import torch
from torch import Tensor


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