import abc
from typing import Callable, Tuple

import torch
from torch import Tensor

from ...irt import DIRT


class Kernel(abc.ABC):

    def __init__(
        self, 
        potential: Callable[[Tensor], Tensor], 
        dirt: DIRT, 
        ys: Tensor | None = None, 
        subset: str = "first"
    ):

        if isinstance(ys, Tensor):
            ys = torch.atleast_2d(ys)
            dim = dirt.dim - ys.shape[1]  # type: ignore
        else:
            dim = dirt.dim

        self.potential = potential
        self.dirt = dirt
        self.ys = ys
        self.subset = subset
        self.reference = dirt.reference
        self.dim = dim
        self.initialised = False
        self.num_steps = 0
        return
    
    @property
    def acceptance_rates(self) -> Tensor:
        return self.num_accepts / self.num_steps
    
    def _out_domain(self, rs: Tensor) -> Tensor:
        """Returns True if a point is outside the support of the 
        reference density, and False otherwise.
        """
        rs = torch.atleast_2d(rs)
        out_domain = self.reference._out_domain(rs).any(dim=1).bool()
        return out_domain
    
    def _initialise(self, r0s: Tensor) -> None:

        r0s = torch.atleast_2d(r0s)
        self.num_chains = r0s.shape[0]
        self.num_accepts = torch.zeros((self.num_chains,))

        self._rs = r0s
        self._xs, self._neglogfrs, self._neglogfxs = self._potential_pull(r0s)
        self.initialised = True
        return
 
    def _potential_pull(self, rs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns the pullback of the target function under the DIRT 
        mapping.
        """

        if self.ys is None:
            rs = torch.atleast_2d(rs)
            return self.dirt.eval_irt_pullback(self.potential, rs, subset=self.subset)
        else:
            rs = torch.atleast_2d(rs)
            return self.dirt.eval_cirt_pullback(self.potential, self.ys, rs, subset=self.subset)

    def _irt_func(self, rs) -> Tensor:
        
        if self.ys is None:
            rs = torch.atleast_2d(rs)
            xs = self.dirt.eval_irt(rs, subset=self.subset)[0]
            return xs
        else:
            rs = torch.atleast_2d(rs)
            xs = self.dirt.eval_cirt(self.ys, rs, subset=self.subset)[0]
            return xs

    @abc.abstractmethod
    def _propose(self) -> Tensor:
        """Proposes a new set of states given the current state of each 
        Markov chain.
        """
        pass

    @abc.abstractmethod 
    def _eval_neglogproposal(self, rs: Tensor, rs_prop: Tensor) -> Tensor:
        """Evaluates the transition kernel.

        Parameters
        ----------
        rs:
            An n * d matrix containing a set of current states.
        rs_prop:
            An n * d matrix containing a set of proposed states.

        Returns
        -------
        neglogproposals:
            An n-dimensional vector containing the negative logarithm 
            of the transition kernel evaluated at each of the sets of 
            states.
            
        """
        pass

    def _step(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Takes a single step.
        
        Returns
        -------

        """

        if not self.initialised:
            msg = "Kernel not initialised."
            raise Exception(msg)
        
        xs_prop = torch.zeros_like(self._rs)
        neglogfrs_prop = torch.zeros_like(self._neglogfrs)
        neglogfxs_prop = torch.zeros_like(self._neglogfxs)

        # Propose a new state for each chain
        rs_prop = self._propose()

        # Check for states outside domain
        out_domain = torch.tensor([self._out_domain(rs) for rs in rs_prop])
        neglogfrs_prop[out_domain] = torch.inf 
        neglogfxs_prop[out_domain] = torch.inf

        # Evaluate the potential of the pullback of the target function 
        # at the proposed states
        (xs_prop[~out_domain], 
         neglogfrs_prop[~out_domain], 
         neglogfxs_prop[~out_domain]) = self._potential_pull(rs_prop[~out_domain])

        neglogqs_prop = self._eval_neglogproposal(self._rs, rs_prop)
        neglogqs_prev = self._eval_neglogproposal(rs_prop, self._rs)
        
        neglogalphas = (
            neglogfrs_prop + neglogqs_prev 
            - (self._neglogfrs + neglogqs_prop)
        )
        alphas = torch.exp(-neglogalphas)
        accepted = alphas > torch.rand_like(alphas)
        
        if accepted.any():
            self._rs[accepted] = rs_prop[accepted]
            self._xs[accepted] = xs_prop[accepted]
            self._neglogfrs[accepted] = neglogfrs_prop[accepted]
            self._neglogfxs[accepted] = neglogfxs_prop[accepted]

        self.num_accepts += accepted.int()
        self.num_steps += 1

        return self._xs, self._neglogfxs, accepted