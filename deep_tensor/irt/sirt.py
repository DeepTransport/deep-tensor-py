import abc
from typing import Callable

import torch

from .abstract_irt import AbstractIRT
from ..approx_bases import ApproxBases
from ..approx_func import ApproxFunc
from ..directions import Direction
from ..input_data import InputData
from ..options import ApproxOptions
from ..polynomials import CDF1D, construct_cdf
from ..tt_data import TTData


class SIRT(AbstractIRT, abc.ABC):
    
    def __init__(
        self, 
        potential: Callable, 
        bases: ApproxBases,
        approx: ApproxFunc|None, 
        options: ApproxOptions, 
        input_data: InputData, 
        approx_data: TTData,
        tau: float
    ):

        AbstractIRT.__init__(
            self,
            potential, 
            bases,
            approx, 
            options, 
            input_data,
            approx_data
        )
        
        self._int_dir = Direction.FORWARD # TEMP??
        self._order = None
        self._tau = tau

        def func(ls: torch.Tensor) -> torch.Tensor:
            """Computes the approximation to the target density for a 
            set of samples in the local domain.
            """
            return self.potential2density(potential, ls)

        self._approx = self.build_approximation(
            func, 
            bases,
            options, 
            input_data,
            approx_data
        )

        self._oned_cdfs = {}
        for k in range(self.bases.dim):
            self._oned_cdfs[k] = construct_cdf(
                poly=self.approx.bases.polys[k], 
                error_tol=self.approx.options.cdf_tol
            )

        self.marginalise()
        return

    @property 
    def oned_cdfs(self) -> dict[int, CDF1D]:
        return self._oned_cdfs

    @property
    def approx(self) -> ApproxFunc:
        return self._approx

    @approx.setter 
    def approx(self, value: ApproxFunc):
        self._approx = value

    @property
    def int_dir(self) -> Direction:
        return self._int_dir
    
    @property
    def order(self) -> torch.Tensor:
        return self._order
    
    @order.setter 
    def order(self, value: torch.Tensor):
        self._order = value
        return

    @property 
    def tau(self) -> torch.Tensor:
        return self._tau
    
    @property 
    def z(self) -> torch.Tensor:
        return self._z 
    
    @property 
    def z_func(self) -> torch.Tensor:
        return self._z_func

    def potential2density(
        self, 
        potential_func: Callable[[torch.Tensor], torch.Tensor], 
        ls: torch.Tensor
    ) -> torch.Tensor:
        """Returns the square root of the target function evaluated at
        a set of samples in the local domain.
        
        Parameters
        ----------
        ls:
            An n * d matrix containing a set of n samples from the 
            local domain ([-1, 1]^d).
        potential_func:
            A function that evaluates the potential (negative log) 
            of the target function at a given set of samples from the 
            approximation domain.

        Returns
        -------
        pls:
            An n-dimensional vector containing the square root of the 
            ratio of the potential function and the weighting function, 
            evaluated at each element of ls.
            
        TODO: check this (and eval_measure_potential) with TC.

        """
        
        xs = self.bases.local2approx(ls)[0]
        neglogfxs = potential_func(xs)
        neglogwxs = self.bases.eval_measure_potential(xs)[0]

        # The ratio of f and w is invariant to changes of coordinate
        pls = torch.exp(-0.5 * (neglogfxs - neglogwxs))
        return pls
    
    def get_potential2density(
        self, 
        ys: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        
        raise Exception("TODO: something seems wrong here.")
        
        _, dxdzs = self.bases.local2approx(ys, zs)

        neglogref = self.bases.eval_measure_potential_local(zs)

        logdet = dxdzs.log().sum(dim=1)
        log_ys = -0.5 * (ys - neglogref - logdet)
        return torch.exp(log_ys)