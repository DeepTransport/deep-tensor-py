import warnings

import torch 
from torch import Tensor


def _next_pow_two(n: int) -> int:
    """Returns the smallest power of two greater than or equal to the 
    input value.
    """
    i = 1
    while i < n:
        i *= 2
    return i


def compute_autocorrelations(xs: Tensor) -> Tensor:
    """Computes the autocorrelations associated with a 1D time series.

    Parameters
    ----------
    xs:
        An n-dimensional vector containing a 1D time series.
    
    Returns
    -------
    acf:
        An n-dimensional vector containing an estimate of the 
        autocorrelations for `xs`.

    References
    ----------
    https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    """
    if xs.dim() != 1:
        raise Exception("Input tensor must be one-dimensional.")
    n = _next_pow_two(xs.numel())
    f = torch.fft.fft(xs - xs.mean(), n=2*n)
    acf = torch.fft.ifft(f * torch.conj(f))[:xs.numel()].real
    acf = acf / acf[0]
    return acf


def estimate_iact(xs: Tensor) -> Tensor:
    """Estimates the integrated autocorrelation time of each parameter 
    within a simulated Markov chain.
    
    Parameters
    ----------
    xs:
        A num_steps * num_params matrix containing the simulated Markov 
        chain.

    Returns
    -------
    taus:
        A vector containing the estimates of the IACT for each 
        parameter.
    
    References
    ----------
    https://mc-stan.org/docs/2_19/reference-manual/effective-sample-size-section.html

    """

    taus = torch.zeros(xs.shape[1], device=xs.device)

    for i, x_i in enumerate(xs.T):
        
        rhos_i = compute_autocorrelations(x_i)
        montone_seq = torch.cummin(rhos_i[:-1:2] + rhos_i[1::2], 0).values

        if montone_seq.min() < 0:
            M = (montone_seq > 0).int().argmin()
        else:
            msg = "Monotone sequence contains no negative component."
            warnings.warn(msg)
            M = montone_seq.numel()
        
        taus[i] = -1.0 + 2.0 * torch.sum(montone_seq[:M])

        # import puwr
        # tau_wolff = puwr.tauint(xs.T[:, None, :].numpy(), i)[2]
    
    return taus