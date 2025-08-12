from typing import Callable, Tuple
import warnings

import torch 
from torch import Tensor

from ..irt import DIRT
from ..references import GaussianReference, Reference


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
        self.xs[self.n_steps] = x_i.flatten()
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
        diagnostics = [
            f"Iteration: {self.n_steps:>5f}", 
            f"Acceptance rate: {self.acceptance_rate:.2f}"
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
    The IACT for parameter $i$, denoted using $\tau_{i}$, is estimated 
    using the monotone sequence estimator outlined by Geyer (2011).

    References
    ----------
    Geyer, CJ (2011). *[Introduction to Markov chain Monte Carlo](https://doi.org/10.1201/b10905)*. 
    In: Handbook of Markov Chain Monte Carlo 3--48.
    
    """
    def __init__(self, chain: MarkovChain):
        self.xs = chain.xs
        self.potentials = chain.potentials
        self.acceptance_rate = chain.acceptance_rate
        self.iacts = estimate_iact(chain.xs)
        self.ess = 1.0 / self.iacts
        return


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

    # Compute the FTT and autocorrelation function
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
        An n_steps * n_params matrix containing the simulated Markov 
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

    taus = torch.zeros(xs.shape[1])

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


def _run_irt_pcn(
    negloglik_pullback: Callable[[Tensor], Tuple[Tensor, Tensor]],
    irt_func: Callable[[Tensor], Tensor],
    reference: Reference,
    dim: int,
    n: int,
    dt: float,
    r0: Tensor | None = None, 
    verbose: bool = True
) -> MCMCResult:
    
    if not isinstance(reference, GaussianReference):
        msg = "DIRT object must have a Gaussian reference density."
        raise Exception(msg)
    
    if dt <= 0.0:
        msg = "Stepsize must be positive."
        raise Exception(msg)

    a = 2.0 * (2.0*dt)**0.5 / (2.0+dt)
    b = (2.0-dt) / (2.0+dt)

    r_c = r0.clone() if r0 is not None else torch.zeros((1, dim))
    x_c = irt_func(r_c)
    negloglik_pull_c, neglogfx_c = negloglik_pullback(r_c)

    chain = MarkovChain(n, dim)
    chain.add_new_state(x_c, neglogfx_c)

    # Sample a set of perturbations and acceptance probabilities
    ps = torch.randn((n-1, dim))
    probs = torch.rand(n-1)

    for i in range(n-1):
        
        # Propose a new state
        r_p = b * r_c + a * ps[i]

        if reference._out_domain(torch.atleast_2d(r_p)).any():
            negloglik_pull_p = torch.tensor(torch.inf)
            neglogfx_p = torch.tensor(torch.inf)
            alpha = -torch.tensor(torch.inf)
        else:
            negloglik_pull_p, neglogfx_p = negloglik_pullback(r_p)
            alpha = negloglik_pull_c - negloglik_pull_p

        if torch.exp(alpha) > probs[i]:
            r_c = r_p.clone()
            x_c = irt_func(r_c)
            negloglik_pull_c = negloglik_pull_p.clone()
            neglogfx_c = neglogfx_p.clone()
            chain.add_new_state(x_c, neglogfx_c)
        else:
            chain.add_current_state()

        if verbose and (i+1) % 100 == 0:
            chain.print_progress()

    return MCMCResult(chain)


def run_irt_pcn(
    potential: Callable[[Tensor], Tensor],
    dirt: DIRT,
    n: int,
    dt: float = 2.0,
    r0: Tensor | None = None,
    subset: str = "first",
    verbose: bool = True
) -> MCMCResult:
    r"""Runs a pCN sampler using the DIRT mapping.
    
    Runs a preconditioned Crank-Nicolson sampler (Cotter *et al.*, 
    2013) to characterise the pullback of the target density under the 
    DIRT mapping, then pushes the resulting samples forward under the 
    DIRT mapping to obtain samples distributed according to the target. 
    This idea was initially outlined by Cui *et al.* (2023).

    Note that the pCN proposal is only applicable to problems with a 
    Gaussian reference density.

    Parameters
    ----------
    potential:
        A function that returns the negative logarithm of the (possibly 
        unnormalised) target density at a given sample.
    dirt:
        A previously-constructed DIRT object.
    n: 
        The length of the Markov chain to construct.
    dt:
        pCN stepsize, $\Delta t$. If this is not specified, a value of 
        $\Delta t = 2$ (independence sampler) will be used.
    r0:
        The starting state. This should be a $1 \times k$ matrix 
        containing a sample from the reference domain. If not passed in, 
        the mean of the reference density will be used.
    subset:
        If the samples contain a subset of the variables, (*i.e.,* 
        $k < d$), whether they correspond to the first $k$ variables 
        (`subset='first'`) or the last $k$ variables (`subset='last'`).
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
        
    dim = dirt.dim
    
    def negloglik_pullback(rs: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the difference between the negative logarithm of the 
        pullback of the target function under the DIRT mapping and the 
        negative log-reference density.
        """
        rs = torch.atleast_2d(rs)
        neglogfr, neglogfx = dirt.eval_irt_pullback(potential, rs, subset=subset)
        neglogref = dirt.reference.eval_potential(rs)[0]
        return neglogfr - neglogref, neglogfx
    
    def irt_func(rs: Tensor) -> Tensor:
        rs = torch.atleast_2d(rs)
        xs = dirt.eval_irt(rs, subset=subset)[0]
        return xs

    res = _run_irt_pcn(
        negloglik_pullback, 
        irt_func, 
        reference=dirt.reference,
        dim=dim,
        n=n, 
        dt=dt,
        r0=r0, 
        verbose=verbose
    )
    return res


def run_cirt_pcn(
    potential: Callable[[Tensor], Tensor],
    dirt: DIRT,
    y: Tensor,
    n: int,
    dt: float = 2.0,
    r0: Tensor | None = None,
    subset: str = "first",
    verbose: bool = True
) -> MCMCResult:
    r"""Runs a pCN sampler using a conditional of the DIRT mapping. 
    
    Runs a pCN sampler to characterise the pullback of the target 
    density under a conditional of the DIRT mapping, then pushes the 
    resulting samples forward under the DIRT mapping to obtain samples 
    distributed according to the target. This idea was initially 
    outlined by Cui *et al.* (2023).

    Note that the pCN proposal is only applicable to problems with a 
    Gaussian reference density.

    Parameters
    ----------
    potential:
        A function that returns the negative logarithm of the (possibly 
        unnormalised) target density at a given sample.
    dirt:
        A previously-constructed DIRT object.
    y:
        A $1 \times k$ matrix containing a sample from the 
        approximation domain to condition on.
    n: 
        The length of the Markov chain to construct.
    dt:
        pCN stepsize, $\Delta t$. If this is not specified, a value of 
        $\Delta t = 2$ (independence sampler) will be used.
    r0:
        The starting state. This should be a $1 \times (d-k)$ matrix 
        containing a sample from the reference domain. If not passed in, 
        the mean of the reference density will be used.
    subset:
        Whether `y` is a realisation of the first $k$ variables 
        (`subset='first'`) or the final $k$ variables (`subset='last'`).
    verbose:
        Whether to print diagnostic information during the sampling 
        process.

    Returns
    -------
    res:
        An object containing the constructed Markov chain and some 
        diagnostic information.

    """

    y = torch.atleast_2d(y)
    dim = dirt.dim - y.shape[1]
    
    def negloglik_pullback(rs: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the difference between the negative logarithm of the 
        pullback of the target function under the DIRT mapping and the 
        negative log-prior density.
        """
        rs = torch.atleast_2d(rs)
        neglogfr, neglogfx = dirt.eval_cirt_pullback(potential, y, rs, subset=subset)
        neglogref = dirt.reference.eval_potential(rs)[0]
        return neglogfr - neglogref, neglogfx

    def irt_func(rs: Tensor) -> Tensor:
        rs = torch.atleast_2d(rs)
        xs = dirt.eval_cirt(y, rs, subset=subset)[0]
        return xs

    res = _run_irt_pcn(
        negloglik_pullback, 
        irt_func, 
        reference=dirt.reference,
        dim=dim,
        n=n, 
        dt=dt,
        r0=r0, 
        verbose=verbose
    )
    return res


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

    n, d = xs.shape
    
    chain = MarkovChain(n, d)
    chain.add_new_state(xs[0], neglogfxs_exact[0])
    i_cur = 0

    for i in range(n-1):

        alpha = (neglogfxs_exact[i_cur] + neglogfxs_irt[i+1]
                 - neglogfxs_exact[i+1] - neglogfxs_irt[i_cur])
        
        if alpha.exp() > torch.rand(1):
            chain.add_new_state(xs[i+1], neglogfxs_exact[i+1])
            i_cur = i+1
        else:
            chain.add_current_state()
    
    return MCMCResult(chain)