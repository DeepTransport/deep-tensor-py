import torch
from torch import Tensor

from .kernel import Kernel
from ..stats import estimate_iact


class MarkovChain(object):
    """Stores a Markov chain constructed by an MCMC sampler.
    
    Parameters
    ----------
    n:
        The final length of the chain.
    dim:
        The dimension of the state space.
    
    """

    def __init__(
        self, 
        num_steps: int, 
        num_chains: int,
        dim: int, 
        device: torch.device
    ):
        self.xs = torch.zeros((num_chains, num_steps, dim), device=device)
        self.potentials = torch.zeros((num_chains, num_steps), device=device)
        self.n = num_steps
        self.num_steps = 0
        self.num_acceptances = torch.zeros((num_chains,), device=device)
        return
    
    @property
    def acceptance_rates(self) -> Tensor:
        return self.num_acceptances / self.num_steps
    
    @property 
    def current_state(self) -> Tensor:
        return self.xs[self.num_steps-1]
    
    @property 
    def current_potential(self) -> Tensor:
        return self.potentials[self.num_steps-1]
    
    def add_state(
        self, 
        xs: Tensor, 
        potentials: Tensor, 
        acceptances: Tensor
    ) -> None:
        """Adds a new state to the end of the Markov chain."""
        self.xs[:, self.num_steps, :] = xs 
        self.potentials[:, self.num_steps] = potentials 
        self.num_acceptances += acceptances
        self.num_steps += 1 
        return
    
    def print_progress(self) -> None:
        diagnostics = [
            f"Iteration: {self.num_steps:>5f}", 
            # f"Acceptance rate: {self.acceptance_rates}"
        ]
        print(" | ".join(diagnostics), end="\r")
        return


class MCMCResult(object):
    r"""An object containing a constructed Markov chain.
    
    Attributes
    ----------
    xs: Tensor
        An $n \times k$ matrix containing the samples that form the 
        Markov chain.
    potentials: Tensor
        An $n$-dimensional vector containing the potential function 
        associated with the target density evaluated at each sample in 
        the chain.
    acceptance_rate: float
        The acceptance rate of the sampler.
    iacts: Tensor
        A $k$-dimensional vector containing estimates of the integrated 
        autocorrelation time (IACT) for each parameter.
    ess: Tensor
        A $k$-dimensional vector containing estimates of the effective 
        sample size (ESS) of each parameter.

    Notes
    -----
    The IACT for each parameter is estimated using the monotone 
    sequence estimator outlined by Geyer (2011).

    References
    ----------
    Geyer, CJ (2011). *[Introduction to Markov chain Monte Carlo](https://doi.org/10.1201/b10905)*. 
    In: Handbook of Markov Chain Monte Carlo 3--48.
    
    """
    def __init__(self, chain: MarkovChain):
        self.num_chains, self.num_steps, self.dim = chain.xs.shape
        self.xs = chain.xs
        self.potentials = chain.potentials
        self.acceptance_rates = chain.acceptance_rates
        self.iacts = torch.vstack([
            estimate_iact(self.xs[i]) for i in range(self.num_chains)
        ])
        self.ess = 1.0 / self.iacts
        return


class MCMC(object):
    """An object used to run an MCMC sampler.
    
    Parameters
    ----------
    kernel: 
        The transition kernel to use.

    """

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        return
    
    @property 
    def acceptance_rates(self) -> Tensor:
        return self.kernel.acceptance_rates
    
    def run(
        self, 
        r0s: Tensor, 
        num_steps: int,
        num_warmup: int = 0
    ):
        """
        TODO: finish this docstring...
        
        r0s:
            An n * d matrix (where n denotes the number of chains to 
            run) containing the starting point for each chain (in the 
            domain of the reference distribution).
        num_steps:
            The number of steps to run each chain for (excluding 
            warm-up steps).
        num_warmup: 
            The number of warmup (also referred to as burn in) steps to 
            take for each chain. These corresponding states are 
            discarded from the results.
        
        """
        
        self.r0s: Tensor = torch.atleast_2d(r0s)
        self.device = self.r0s.device
        self.num_chains = self.r0s.shape[0]
        self.num_steps = num_steps
        self.num_warmup = num_warmup

        self.kernel._initialise(self.r0s)
        
        for _ in range(self.num_warmup):
            self.kernel._step()

        self.chain = MarkovChain(
            self.num_steps, 
            self.num_chains, 
            self.kernel.dim, 
            device=self.device
        )
        
        for _ in range(self.num_steps):
            xs, potentials, acceptances = self.kernel._step()
            self.chain.add_state(xs, potentials, acceptances)

        res = MCMCResult(self.chain)
        return res