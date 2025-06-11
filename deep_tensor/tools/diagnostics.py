import torch 
from torch import Tensor

import warnings


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
        particless.

    References
    ----------
    Owen, AB (2013). Monte Carlo theory, methods and examples. Chapter 9.

    """

    sample_size = log_weights.numel()
    log_weights = log_weights - log_weights.max()

    ess = log_weights.exp().sum().square() / (2.0*log_weights).exp().sum()
    ess_ratio = ess / sample_size
    return ess_ratio


def _next_pow_two(n: int) -> int:
    """Returns the smallest power of two greater than or equal to the 
    input value.
    """
    i = 1
    while i < n:
        i *= 2
    return i


def _autocorr_1d(xs: Tensor) -> Tensor:
    """Computes the autocorrelations associated with a 1D time series.

    https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    """

    if xs.dim() != 1:
        raise Exception("Input tensor must be one-dimensional.")

    # Compute the FTT and autocorrelation function
    n = _next_pow_two(xs.numel())
    f = torch.fft.fft(xs - xs.mean(), n=2*n)
    acf = torch.fft.ifft(f * torch.conj(f))[:xs.numel()].real
    acf = acf / acf[0]
    return acf


def _estimate_window(taus: Tensor, c: float) -> int:
    """Computes a suitable window size to use when estimating the IACT.
    """
    ms = torch.arange(taus.numel()) > c * taus
    if torch.any(ms):
        return int(torch.nonzero(ms)[0])
    warnings.warn("Could not find a suitable window size.")
    return taus.numel() - 1


def estimate_iact(xs: Tensor, c: float = 5.0) -> Tensor:
    """Estimates the integrated autocorrelation time of a simulated 
    Markov chain.
    
    Parameters
    ----------
    xs:
        An n_steps * n_params matrix containing the simulated Markov 
        chain.
    c:
        Parameter used to determine the window size to use when 
        estimating the IACT.

    Returns
    -------
    taus:
        A vector containing the estimates of the IACT for each 
        parameter.
    
    References
    ----------
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#computing-autocorrelation-times

    """

    taus = torch.zeros(xs.shape[1])

    for i, x_i in enumerate(xs.T):
        rhos_i = _autocorr_1d(x_i)[1:]  # Remove rho(0) = 1
        taus_i = 1.0 + 2.0 * rhos_i.cumsum(dim=0)
        M = _estimate_window(taus_i, c)
        taus[i] = taus_i[M]
    
    return taus