import math
from typing import Callable
import warnings

import torch 
from torch import Tensor

from .kernel import Kernel
from ...irt import DIRT
from ...references import GaussianReference


class pCNKernel(Kernel):
    r"""The preconditioned Crank-Nicolson proposal (Cotter *et al.*, 2013).

    Parameters
    ----------
    potential:
        A function that returns the negative logarithm of the (possibly 
        unnormalised) target density at a given sample.
    dirt:
        A previously-constructed DIRT object.
    subset:
        If the samples contain a subset of the variables, (*i.e.,* 
        $k < d$), whether they correspond to the first $k$ variables 
        (`subset='first'`) or the last $k$ variables (`subset='last'`).
    dt:
        pCN stepsize, $\Delta t$. If this is not specified, a value of 
        $\Delta t = 2$ (independence sampler) will be used.

    Returns
    -------
    res:
        An object containing the constructed Markov chain and some 
        diagnostic information.

    Notes
    -----
    Note that the pCN proposal is only applicable to problems with a 
    standard Gaussian reference density (that is, 
    $\rho(\theta) = \mathcal{N}(0_{d}, I_{d})$). The pCN proposal 
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

    """

    def __init__(
        self, 
        potential: Callable[[Tensor], Tensor], 
        dirt: DIRT, 
        ys: Tensor | None = None,
        subset: str = "first",
        dt: float = 10.0
    ):
        
        if not isinstance(dirt.reference, GaussianReference):
            msg = "The pCN kernel requires a Gaussian reference density."
            raise Exception(msg)
        
        if dt <= 0.0:
            msg = "Stepsize must be positive."
            raise Exception(msg)
        
        if dt == 2.0:
            msg = (
                "Setting dt=2.0 in the pCN kernel results in an " 
                "independence sampler. It is more efficient to use "
                "the dedicated independence sampling function."
            )
            warnings.warn(msg)

        self.a = 2.0 * math.sqrt(2.0*dt) / (2.0+dt)
        self.b = (2.0-dt) / (2.0+dt)

        Kernel.__init__(self, potential, dirt, ys, subset)
        return
    
    def _propose(self) -> Tensor:
        xis = torch.randn((self.num_chains, self.dim))
        rs_prop = self.b * self._rs + self.a * xis
        return rs_prop
    
    def _eval_neglogproposal(self, rs: Tensor, rs_prop: Tensor) -> Tensor:
        # TODO: could test this function
        mus = self.b * rs
        neglogproposals = (
            0.5 * self.dim * math.log(2.0*math.pi)
            + self.dim * self.a
            + (1.0 / (2.0*self.a**2)) * (rs_prop - mus).square().sum(dim=1)
        )
        return neglogproposals