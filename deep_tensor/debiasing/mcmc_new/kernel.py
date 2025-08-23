import abc
from typing import Callable, Tuple

import torch
from torch import Tensor

from ...irt import DIRT


# TODO: could support an n_layers parameter (sometimes it could be a 
# good idea to use only a subset of the layers of the DIRT mapping).


class Kernel(abc.ABC):

    def __init__(
        self, 
        potential: Callable[[Tensor], Tensor], 
        dirt: DIRT, 
        ys: Tensor | None = None, 
        subset: str = "first"
    ):
        """TODO: docstring.
        """
        
        if ys is None:
            dim = dirt.dim    
        else:
            ys = torch.atleast_2d(ys)
            dim = dirt.dim - ys.shape[1]

        self.potential = potential
        self.dirt = dirt
        self.ys = ys
        self.subset = subset
        self.reference = dirt.reference
        self.dim = dim
        self.initialised = False
        self.n_steps = 0
        self.n_accept = 0
        return
    
    @property
    def acceptance_rate(self) -> float:
        return self.n_accept / self.n_steps
    
    def _out_domain(self, rs: Tensor) -> bool:
        """Returns True if a point is outside the support of the 
        reference density, and False otherwise.
        """
        rs = torch.atleast_2d(rs)
        out_domain = bool(self.reference._out_domain(rs).any())
        return out_domain
    
    def _initialise(self, r0: Tensor | None = None) -> None:

        if r0 is None:
            r0 = self.reference.random(d=self.dim, n=1)
            x0 = self._irt_func(r0)
        else:
            # TODO: confirm that r0 has the same length as the 
            # dimension of the kernel.
            raise NotImplementedError()
            r0 = torch.atleast_2d(r0)
            x0 = self.irt_func(r0)

        self._r_prev = r0
        self._x_prev = x0
        _neglogfr_prev, self._neglogfx_prev = self._potential_pull(r0)
        _neglogref_prev = self.reference.eval_potential(r0)[0]
        self._negloglik_prev = _neglogfr_prev - _neglogref_prev

        self.initialised = True
        return
 
    def _potential_pull(self, rs: Tensor) -> Tuple[Tensor, Tensor]:
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
    def _step(self) -> Tuple[Tensor, Tensor]:
        """Takes a single step, and returns the resulting state and 
        potential associated with the state.
        """
        pass