import torch 
from torch import Tensor

from .mcmc import MarkovChain, MCMCResult


def run_independence_sampler(
    xs: Tensor,
    neglogfxs_irt: Tensor,
    neglogfxs_exact: Tensor
) -> MCMCResult:
    r"""Runs an independence Metropolis-Hastings sampler.
    
    Runs an independence Metropolis-Hastings sampler which uses a dirt 
    density as a proposal.

    Parameters
    ----------
    xs:
        An $n \times d$ matrix containing independent samples from the 
        DIRT object.
    neglogfxs_irt:
        An $n$-dimensional vector containing the potential function 
        associated with the DIRT object evaluated at each sample.
    neglogfxs_exact:
        An $n$-dimensional vector containing the potential function 
        associated with the target density evaluated at each sample.

    Returns
    -------
    res:
        An object containing the constructed Markov chain and some 
        diagnostic information.
    
    """

    num_steps, d = xs.shape
    num_chains = 1
    
    acceptances = torch.tensor([0], device=xs.device)
    chain = MarkovChain(num_steps, num_chains, d, device=xs.device)
    chain.add_state(xs[0], neglogfxs_exact[0], acceptances)
    i_cur = 0

    for i in range(num_steps-1):

        alpha = (neglogfxs_exact[i_cur] + neglogfxs_irt[i+1]
                 - neglogfxs_exact[i+1] - neglogfxs_irt[i_cur])
        
        acceptances = alpha.exp() > torch.rand(num_chains, device=xs.device)
        if acceptances:
            chain.add_state(xs[i+1], neglogfxs_exact[i+1], acceptances)
            i_cur = i+1
        else:
            chain.add_state(xs[i_cur], neglogfxs_exact[i_cur], acceptances)
    
    return MCMCResult(chain)