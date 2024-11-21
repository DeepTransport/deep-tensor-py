import abc
from typing import Callable

import torch

from .abstract_irt import AbstractIRT
from ..approx_bases import ApproxBases
from ..approx_func import ApproxFunc
from ..directions import Direction
from ..input_data import InputData
from ..options import ApproxOptions
from ..polynomials import OnedCDF, construct_cdf


class SIRT(AbstractIRT, abc.ABC):
    
    def __init__(
        self, 
        potential: Callable, 
        bases: ApproxBases, 
        options: ApproxOptions, 
        input_data: InputData, 
        tau: float
    ):

        # TODO: maybe some of this stuff should be part of AbstractIRT
        self.potential = potential 
        self.bases = bases 
        self.options = options 
        self.input_data = input_data 
        
        self._int_dir = Direction.FORWARD
        self._tau = tau

        def func(z: torch.Tensor) -> torch.Tensor:
            return self.potential2density(potential, z)

        self._approx = self.build_approximation(func, bases, options, input_data)

        self._oned_cdfs = {}
        for k in range(self.bases.dim):
            self._oned_cdfs[k] = construct_cdf(
                poly=self.approx.bases.polys[k], 
                error_tol=self.approx.options.cdf_tol
            )

        self._order = None

        self.marginalise()
        return

    @property 
    def oned_cdfs(self) -> dict[int, OnedCDF]:
        return self._oned_cdfs

    @property
    def approx(self) -> ApproxFunc:
        return self._approx

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
        potential_func: Callable, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        """Computes the density of a sample, or set of samples, from 
        the reference domain.

        Parameters
        ----------
        potential_func:
            A function that returns the potential of the target density 
            function at a given point in the approximation domain.
        zs:
            A set of samples from the reference domain, of dimension 
            n * d.

        Returns
        ------
        ys:
            An n-dimensional vector containing the square root of the 
            (unnormalised) target density function evaluated at each 
            (transformed) value of zs.
            
        TODO: I think this returns g(x) (i.e. square root of pi -- see 
        Cui and Dolgov, Eq. 18).
        TODO: figure out what the reference function is doing here.
        """
        
        xs, dxdzs = self.bases.reference2domain(zs)
        ys = potential_func(xs)
        
        neglogref = self.bases.eval_measure_potential_reference(zs)

        logdet = torch.sum(torch.log(dxdzs), 1)
        ys = torch.exp(-0.5*(ys-neglogref-logdet))
        return ys
    
    def get_potential2density(
        self, 
        ys: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        """Computes the density of a sample, or set of samples, from 
        the reference domain.

        Parameters
        ----------
        ys:
            The potential function associated with the target density, 
            evaluated at each (transformed) sample from zs.
        zs:
            A set of samples from the reference domain, of dimension 
            n * d.

        Returns
        ------
        ys:
            An n-dimensional vector containing the square root of the 
            (unnormalised) target density function evaluated at each 
            (transformed) value of zs.
        
        """
        
        _, dxdzs = self.bases.reference2domain(ys, zs)

        neglogref = self.bases.eval_measure_potential_reference(zs)

        logdet = torch.sum(torch.log(dxdzs), 1)
        ys = torch.exp(-0.5*(ys-neglogref-logdet))
        return ys