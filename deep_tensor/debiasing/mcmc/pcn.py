import math
from typing import Callable
import warnings

import torch 
from torch import Tensor

from .kernel import Kernel
from ...irt import DIRT
from ...references import GaussianReference


class pCNKernel(Kernel):
    r"""The preconditioned Crank-Nicolson proposal.

    Parameters
    ----------
    potential:
        A function that returns the negative logarithm of the (possibly 
        unnormalised) target density at a given sample.
    dirt:
        A previously-constructed DIRT object.
    ys:
        TODO: finish this.
    subset:
        If the samples contain a subset of the variables, (*i.e.,* 
        $k < d$), whether they correspond to the first $k$ variables 
        (`subset='first'`) or the last $k$ variables (`subset='last'`).
    dt:
        pCN stepsize, $\Delta t$. If this is not specified, a value of 
        $\Delta t = 2$ (independence sampler) will be used.

    Notes
    -----
    Note that the pCN proposal is only applicable to problems with a 
    standard Gaussian reference density (that is, 
    $\rho(\theta) = \mathcal{N}(0_{d}, I_{d})$). The pCN proposal 
    (given current state $\theta^{(i)}$) takes the form [@Cotter2013]
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

        self._a = 2.0 * math.sqrt(2.0*dt) / (2.0+dt)
        self._b = (2.0-dt) / (2.0+dt)

        Kernel.__init__(self, potential, dirt, ys, subset)
        return
    
    def _propose(self) -> Tensor:
        xis = torch.randn((self._num_chains, self._dim))
        rs_prop = self._b * self._rs + self._a * xis
        return rs_prop
    
    def _eval_neglogproposal(self, rs: Tensor, rs_prop: Tensor) -> Tensor:
        # TODO: could test this function
        mus = self._b * rs
        neglogproposals = (
            0.5 * self._dim * math.log(2.0*math.pi)
            + self._dim * self._a
            + (1.0 / (2.0*self._a**2)) * (rs_prop - mus).square().sum(dim=1)
        )
        return neglogproposals