from torch import Tensor


def compute_ess_ratio(log_weights: Tensor) -> Tensor:
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