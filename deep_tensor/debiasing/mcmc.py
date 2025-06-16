from typing import Callable

import torch 
from torch import Tensor

from ..irt import AbstractDIRT
from ..references import GaussianReference
from ..tools import estimate_iact


class MarkovChain(object):
    """Stores a Markov chain constructed by an MCMC sampler.
    
    Parameters
    ----------
    n:
        The final length of the chain.
    dim:
        The dimension of the state space.
    
    """

    def __init__(self, n: int, dim: int):
        self.xs = torch.zeros((n, dim))
        self.potentials = torch.zeros(n)
        self.n = n
        self.n_steps = 0
        self.n_accept = 0
        return
    
    @property
    def acceptance_rate(self) -> float:
        return self.n_accept / self.n_steps
    
    @property 
    def current_state(self) -> Tensor:
        return self.xs[self.n_steps-1]
    
    @property 
    def current_potential(self) -> Tensor:
        return self.potentials[self.n_steps-1]
    
    def add_new_state(self, x_i: Tensor, potential_i: Tensor) -> None:
        """Adds a new state to the end of the Markov chain."""
        self.xs[self.n_steps] = x_i
        self.potentials[self.n_steps] = potential_i
        self.n_steps += 1
        self.n_accept += 1
        return
    
    def add_current_state(self) -> None:
        """Adds the current state to the end of the Markov chain."""
        self.xs[self.n_steps] = self.current_state
        self.potentials[self.n_steps] = self.current_potential
        self.n_steps += 1
        return
    
    def print_progress(self) -> None:
        # TODO: finish this.
        # print(self.acceptance_rate)
        return


class MCMCResult(object):
    r"""An object containing a constructed Markov chain.
    
    Parameters
    ----------
    xs:
        An $n \times d$ matrix containing the samples that form the 
        Markov chain.
    acceptance_rate:
        The acceptance rate of the sampler.
    
    """
    def __init__(self, chain: MarkovChain):
        self.xs = chain.xs
        self.potentials = chain.potentials
        self.acceptance_rate = chain.acceptance_rate
        self.iacts = estimate_iact(chain.xs)
        # import puwr
        # print(2.0 * puwr.tauint(chain.xs.T[:, None, :].numpy(), 0)[2])
        return


def _run_irt_pcn(
    negloglik_pullback: Callable[[Tensor], Tensor],
    irt_func: Callable[[Tensor], Tensor],
    reference: GaussianReference,
    dim: int,
    n: int,
    dt: float,
    x0: Tensor | None = None, 
    verbose: bool = True
) -> MCMCResult:

    r0 = irt_func(x0) if x0 is not None else torch.zeros((1, dim))
    negloglik0 = negloglik_pullback(r0)

    a = 2.0 * (2.0*dt)**0.5 / (2.0+dt)
    b = (2.0-dt) / (2.0+dt)

    chain = MarkovChain(n, dim)
    chain.add_new_state(r0, negloglik0)

    # Sample a set of perturbations
    ps = torch.randn((n, dim))

    for i in range(n-1):
        
        # Propose a new state
        r_p = b * chain.current_state + a * ps[i]

        if reference._out_domain(torch.atleast_2d(r_p)).any():
            negloglik_p = torch.tensor(torch.inf)
            alpha = -torch.tensor(torch.inf)
        else:
            negloglik_p = negloglik_pullback(r_p)
            alpha = chain.current_potential - negloglik_p

        if alpha.exp() > torch.rand(1):
            chain.add_new_state(r_p, negloglik_p)
        else:
            chain.add_current_state()

        if verbose and (i+1) % 100 == 0:
            chain.print_progress()

    return MCMCResult(chain)


def run_dirt_pcn(
    potential: Callable[[Tensor], Tensor],
    dirt: AbstractDIRT,
    n: int,
    dt: float = 2.0,
    y_obs: Tensor | None = None,
    x0: Tensor | None = None,
    subset: str = "first",
    verbose: bool = True
) -> MCMCResult:
    r"""Runs a preconditioned Crank-Nicholson (pCN) sampler.
    
    Runs a pCN sampler (Cotter *et al.*, 2013) to characterise the 
    pullback of the target density under the DIRT mapping, then pushes 
    the resulting samples forward under the DIRT mapping to obtain 
    samples distributed according to the target. This idea was 
    initially outlined by Cui *et al.* (2023).

    Note that the pCN proposal is only applicable to problems with a 
    Gaussian reference density.

    Parameters
    ----------
    potential:
        A function that returns the negative logarithm of the (possibly 
        unnormalised) target density at a given sample.
    dirt:
        A previously-constructed DIRT object.
    y_obs:
        A tensor containing the observations.
    n: 
        The length of the Markov chain to construct.
    dt:
        pCN stepsize, $\Delta t$. If this is not specified, a value of 
        $\Delta t = 2$ (independence sampler) will be used.
    x0:
        The starting state. If this is passed in, the DIRT mapping will 
        be applied to it to generate the starting location for sampling 
        from the pullback of the target density. Otherwise, the mean of 
        the reference density will be used.
    verbose:
        Whether to print diagnostic information during the sampling 
        process.

    Returns
    -------
    res:
        An object containing the constructed Markov chain and some 
        diagnostic information.

    Notes
    -----
    When the reference density is the standard Gaussian density (that 
    is, $\rho(\theta) = \mathcal{N}(0_{d}, I_{d})$), the pCN proposal 
    (given current state $\theta^{(i)}$) takes the form
    $$
        \theta' = \frac{2-\Delta t}{2+\Delta t} \theta^{(i)} 
            + \frac{2\sqrt{2\Delta t}}{2 + \Delta t} \tilde{\theta},
    $$
    where $\tilde{\theta} \sim \rho(\,\cdot\,)$, and $\Delta t$ denotes 
    the step size. 

    When $\Delta t = 2$, the resulting sampler is an independence 
    sampler. When $\Delta t > 2$, the proposals are negatively 
    correlated, and when $\Delta t < 2$, the proposals are positively 
    correlated.

    References
    ----------
    Cotter, SL, Roberts, GO, Stuart, AM and White, D (2013). *[MCMC 
    methods for functions: Modifying old algorithms to make them 
    faster](https://doi.org/10.1214/13-STS421).* Statistical Science 
    **28**, 424--446.

    Cui, T, Dolgov, S and Zahm, O (2023). *[Scalable conditional deep 
    inverse Rosenblatt transports using tensor trains and gradient-based 
    dimension reduction](https://doi.org/10.1016/j.jcp.2023.112103).* 
    Journal of Computational Physics **485**, 112103.

    """

    if not isinstance(dirt.reference, GaussianReference):
        msg = "DIRT object must have a Gaussian reference density."
        raise Exception(msg)
    
    if dt <= 0.0:
        msg = "Stepsize must be positive."
        raise Exception(msg)
    
    if y_obs is not None:

        y_obs = torch.atleast_2d(y_obs)
        dim = dirt.dim - y_obs.shape[1]
        
        def negloglik_pullback(rs: Tensor) -> Tensor:
            """Returns the difference between the negative logarithm of the 
            pullback of the target function under the DIRT mapping and the 
            negative log-prior density.
            """
            rs = torch.atleast_2d(rs)
            neglogfr = dirt.eval_cirt_pullback(potential, y_obs, rs, subset=subset)
            neglogref = dirt.reference.eval_potential(rs)[0]
            return neglogfr - neglogref
    
        def irt_func(rs: Tensor) -> Tensor:
            rs = torch.atleast_2d(rs)
            ms = dirt.eval_cirt(y_obs, rs, subset=subset)[0]
            return ms
        
    else:

        dim = dirt.dim
        
        def negloglik_pullback(rs: Tensor) -> Tensor:
            """Returns the difference between the negative logarithm of the 
            pullback of the target function under the DIRT mapping and the 
            negative log-prior density.
            """
            rs = torch.atleast_2d(rs)
            neglogfr = dirt.eval_irt_pullback(potential, rs, subset=subset)
            neglogref = dirt.reference.eval_potential(rs)[0]
            return neglogfr - neglogref
        
        def irt_func(rs: Tensor) -> Tensor:
            rs = torch.atleast_2d(rs)
            ms = dirt.eval_irt(rs, subset=subset)[0]
            return ms

    res = _run_irt_pcn(
        negloglik_pullback, 
        irt_func, 
        reference=dirt.reference,
        dim=dim,
        n=n, 
        dt=dt,
        x0=x0, 
        verbose=verbose
    )
    return res


def run_independence_sampler(
    xs: Tensor,
    neglogfxs_irt: Tensor,
    neglogfxs_exact: Tensor
) -> MCMCResult:
    r"""Runs an independence MCMC sampler.
    
    Runs an independence MCMC sampler using a set of samples from a 
    SIRT or DIRT object as the proposal.

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

    n, d = xs.shape
    potentials = neglogfxs_exact - neglogfxs_irt
    
    chain = MarkovChain(n, d)
    chain.add_new_state(xs[0], potentials[0])

    for i in range(n-1):
        
        alpha = chain.current_potential - potentials[i+1]
        
        if alpha.exp() > torch.rand(1):
            chain.add_new_state(xs[i+1], potentials[i+1])
        else:
            chain.add_current_state()
    
    return MCMCResult(chain)