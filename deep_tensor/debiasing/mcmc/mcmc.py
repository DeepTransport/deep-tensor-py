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
        self._xs = torch.zeros((num_chains, num_steps, dim), device=device)
        self._potentials = torch.zeros((num_chains, num_steps), device=device)
        self._n = num_steps
        self._num_steps = 0
        self._num_acceptances = torch.zeros((num_chains,), device=device)
        return
    
    @property
    def _acceptance_rates(self) -> Tensor:
        return self._num_acceptances / self._num_steps
    
    @property 
    def _current_state(self) -> Tensor:
        return self._xs[self._num_steps-1]
    
    @property 
    def _current_potential(self) -> Tensor:
        return self._potentials[self._num_steps-1]
    
    def _add_state(
        self, 
        xs: Tensor, 
        potentials: Tensor, 
        acceptances: Tensor
    ) -> None:
        """Adds a new state to the end of the Markov chain."""
        self._xs[:, self._num_steps, :] = xs 
        self._potentials[:, self._num_steps] = potentials 
        self._num_acceptances += acceptances
        self._num_steps += 1 
        return
    
    def _print_progress(self) -> None:
        diagnostics = [
            f"Iteration: {self._num_steps:>5f}", 
            # f"Acceptance rate: {self._acceptance_rates}"
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
    acceptance_rates: Tensor
        The acceptance rate of each chain.
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
        self._num_chains, self._num_steps, self._dim = chain._xs.shape
        self.xs = chain._xs
        self.potentials = chain._potentials
        self.acceptance_rates = chain._acceptance_rates
        self.iacts = torch.vstack([
            estimate_iact(self.xs[i]) for i in range(self._num_chains)
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
        self._kernel = kernel
        return
    
    @property 
    def _acceptance_rates(self) -> Tensor:
        return self._kernel._acceptance_rates
    
    def run(
        self, 
        r0s: Tensor, 
        num_steps: int,
        num_warmup: int = 0
    ):
        r"""Runs the MCMC sampler.
        
        Parameters
        ----------
        r0s:
            An $n \times d$ matrix (where n denotes the number of 
            chains to run) containing the starting point for each chain 
            (in the domain of the reference distribution).
        num_steps:
            The number of steps to run each chain for (excluding 
            warm-up steps).
        num_warmup: 
            The number of warmup (also referred to as burn-in) steps to 
            take for each chain. These corresponding states are 
            discarded from the results.
        
        """
        
        self._r0s: Tensor = torch.atleast_2d(r0s)
        self._device = self._r0s.device
        self._num_chains = self._r0s.shape[0]
        self._num_steps = num_steps
        self._num_warmup = num_warmup

        self._kernel._initialise(self._r0s)
        
        for _ in range(self._num_warmup):
            self._kernel._step()

        self._chain = MarkovChain(
            self._num_steps, 
            self._num_chains, 
            self._kernel._dim, 
            device=self._device
        )
        
        for _ in range(self._num_steps):
            xs, potentials, acceptances = self._kernel._step()
            self._chain._add_state(xs, potentials, acceptances)

        res = MCMCResult(self._chain)
        return res