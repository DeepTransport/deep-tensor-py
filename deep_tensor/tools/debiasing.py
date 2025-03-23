import torch 
from torch import Tensor


def run_mcmc(
    xs: Tensor,
    neglogfxs_irt: Tensor,
    neglogfxs_exact: Tensor
) -> Tensor:
    r"""Runs an independence MCMC sampler.
    
    Runs an independence MCMC sampler using a set of samples from a 
    SIRT or DIRT object as the proposal.

    Parameters
    ----------
    xs:
        An $n \times d$ matrix containing independent samples from the 
        SIRT or DIRT object.
    neglogfxs_irt:
        An $n$-dimensional vector containing the potential function 
        associated with the SIRT or DIRT object evaluated at each 
        sample.
    neglogfxs_exact:
        An $n$-dimensional vector containing the potential function 
        associated with the target density evaluated at each sample.

    Returns
    -------
    xs_chain: 
        An $n \times d$ matrix containing the samples that are part of 
        the constructed Markov chain.
    
    """

    xs_chain = xs.clone()
    neglogfxs_e = neglogfxs_exact.clone()
    neglogfxs_i = neglogfxs_irt.clone()
    
    n_xs = xs.shape[0]
    n_accept = 0

    for i in range(n_xs-1):
        
        alpha = (neglogfxs_i[i+1] - neglogfxs_e[i+1] 
                    - neglogfxs_i[i] + neglogfxs_e[i])
        
        if alpha.exp() < torch.rand(1):
            xs_chain[i+1] = xs_chain[i]
            neglogfxs_i[i+1] = neglogfxs_i[i]
            neglogfxs_e[i+1] = neglogfxs_e[i]
        else:
            n_accept += 1

    print(f"MCMC complete. Acceptance rate: {n_accept/n_xs:.4f}.")
    return xs_chain