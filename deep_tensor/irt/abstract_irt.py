import abc
from typing import Callable, Tuple

import torch

from ..approx_bases import ApproxBases
from ..approx_func import ApproxFunc
from ..constants import EPS
from ..directions import Direction
from ..input_data import InputData
from ..options import ApproxOptions
from ..polynomials import CDF1D


Z_MIN = torch.tensor(EPS)
Z_MAX = torch.tensor(1.0-EPS)


class AbstractIRT(abc.ABC):
    """TODO: write docstring for this."""

    @property
    @abc.abstractmethod 
    def approx(self) -> ApproxFunc:
        """The approximation of the square root of the target density.
        """
        return

    @property
    @abc.abstractmethod  
    def tau(self) -> torch.Tensor:
        """The defensive term (used to ensure that the tails of the 
        approximation are sufficiently heavy).
        """
        return 

    @property
    @abc.abstractmethod 
    def z_func(self) -> torch.Tensor:
        """The normalising constant of the function approximation part 
        of the target density.
        """
        return 
    
    @property
    @abc.abstractmethod  
    def z(self) -> torch.Tensor:
        """The normalising constant associated with the approximation 
        to the target density.
        """
        return 
    
    @property
    @abc.abstractmethod  
    def oned_cdfs(self) -> dict[int, CDF1D]:
        """One-dimensional polynomial bases for building the CDF of the
        approximation to the target distribution.
        """
        return

    @property 
    @abc.abstractmethod
    def order(self) -> torch.Tensor:
        """The order in which to compute the marginals in each 
        dimension.
        """
        return

    @property
    @abc.abstractmethod
    def int_dir(self) -> Direction:
        """The direction in which to integrate over the approximation.
        """
        return

    @abc.abstractmethod
    def potential2density(
        self,
        bases: ApproxBases, 
        func: Callable,
        zs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return
    
    @abc.abstractmethod
    def get_potential2density(
        self,
        ys: torch.Tensor,
        zs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return

    @abc.abstractmethod
    def build_approximation(
        self, 
        func: Callable, 
        bases: ApproxBases, 
        options: ApproxOptions,
        input_data: InputData,
    ) -> ApproxFunc:
        """Constructs a functional approximation to a given target 
        density function.

        Parameters
        ----------
        func:
            A function that returns the square root of the target 
            density for a sample from the reference domain.
        bases:
            The (polynomial) basis associated with each dimension.
        options:
            Options used when constructing the approximation to the 
            target density function.
        input_data:
            TODO:

        Returns
        -------
        approx:
            The functional approximation to the target density.

        """
        return

    @abc.abstractmethod 
    def eval_potential_reference(
        self, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return

    @abc.abstractmethod
    def eval_rt_reference(
        self, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return

    @abc.abstractmethod 
    def eval_rt_jac_reference(
        self,
        zs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return

    @abc.abstractmethod
    def eval_irt_reference_nograd(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: write docstring for this."""
        return

    @abc.abstractmethod 
    def eval_cirt_reference(
        self, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return

    @abc.abstractmethod 
    def marginalise(self, dir) -> None:
        """Marginalises the approximation to the target function.
        
        Parameters
        ----------
        dir:
            The direction in which to compute the marginalisation 
            (i.e., whether the marginalise from the first to the last 
            dimension or the last to the first).

        Returns
        -------
        None 
            
        """
        return

    def get_transform_indices(self, dim_z: int) -> torch.Tensor:
        """TODO: write docstring."""
        # TODO: I think there will be issues with the below (currently 
        # there is no way to specify that int_dir is not FORWARD or BACKWARD)

        if self.int_dir == Direction.FORWARD:
            return torch.arange(dim_z)
        elif self.int_dir == Direction.BACKWARD:
            return torch.arange(self.approx.dim-dim_z, self.approx.dim)
        elif self.order.numel() != 0:
            return self.order[:dim_z]
        
        msg = "Either `order` or `int_dir` must be specified."
        raise Exception(msg)

    def eval_potential(self, xs: torch.Tensor):
        """Evaluates the (normalised) marginal potential function.

        Parameters
        ----------
        xs:
            An n * d matrix containing samples from the reference 
            domain.
        
        Returns
        -------
        fxs:
            The marginal density evaluated at each sample contained in 
            xs.

        """

        dim_z = xs.shape[1]
        indices = self.get_transform_indices(dim_z)
        
        rs, drdxs = self.approx.bases.domain2reference(xs, indices)
        frs = self.eval_potential_reference(rs)
        # TODO: check whether the sum direction is correct.
        fxs = frs - torch.sum(torch.log(drdxs), 1)
        return fxs

    def eval_pdf(
        self, 
        xs: torch.Tensor 
    ) -> torch.Tensor: 
        """Evaluates the normalised marginal PDF at a given set of x 
        values.
        
        TODO: finish
        """

        fxs = self.eval_potential(xs)
        fxs = torch.exp(-fxs)
        return fxs
    
    def eval_rt(self, xs: torch.Tensor) -> torch.Tensor:
        """Evaluates the squared Rosenblatt transport Z = R(X), where 
        Z is a (unit) uniform random variable and X is the target 
        random variable.

        Parameters
        ----------
        xs: 
            A set of realisations of X.
        
        Returns
        -------
        :
            The corresponding realisations of Z.
        
        """

        dim_x = xs.shape[1]
        indices = self.get_transform_indices(dim_x)
        
        rs, _ = self.approx.bases.domain2reference(xs, indices)
        zs = self.eval_rt_reference(rs)
        return zs
    
    def eval_rt_jac(
        self, 
        xs: torch.Tensor,
        zs: torch.Tensor 
    ) -> torch.Tensor:
        """Evaluates the Jacobian of the squared Rosenblatt transport 
        Z = R(X), where Z is the uniform random variable and X is the 
        target random variable.

        TODO: finish
        """
        raise NotImplementedError()

    def eval_irt_nograd(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given a set of samples of a standard uniform random 
        variable, Z, computes the corresponding samples from the 
        approximation to the target PDF by applying the inverse 
        Rosenblatt transport.
        
        Parameters
        ----------
        zs: 
            An n * d matrix containing uniformly distributed samples.
        
        Returns
        -------
        xs: 
            An n * d matrix containing the corresponding samples from 
            the PDF defined by SIRT.
        potential_xs: 
            An n-dimensional vector containing the potential function
            associated with the target density evaluated at each sample
            in xs.
        
        """

        # TEMP (TODO: remove)
        self._int_dir = Direction.FORWARD

        zs = torch.clamp(zs, Z_MIN, Z_MAX)
        indices = self.get_transform_indices(zs.shape[1])

        rs, potential_rs = self.eval_irt_reference_nograd(zs)
        xs, dxdrs = self.approx.bases.reference2domain(rs, indices)
        potential_xs = potential_rs + dxdrs.log().sum(dim=1)

        return xs, potential_xs
    
    def eval_cirt(
        self, 
        xs: torch.Tensor,
        zs: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the inverse of ..."""

        raise NotImplementedError()
    
    def random(self, n: int) -> torch.Tensor: 
        """Generates a set of random samples from the approximation to
        the target density function, by first sampling a set of 
        independent uniform variates and random, then applying the 
        inverse Rosenblatt transport.
        
        Parameters
        ----------
        n:  
            The number of samples to generate.

        Returns
        -------
        rs:
            The generated samples.
        
        """

        us = torch.rand(n, self.approx.bases.dim)
        rs = self.eval_irt(us)
        return rs 
    
    def sobol(self, n: int) -> torch.Tensor:
        """Generates a set of QMC samples from the approximation to the 
        target density function using a Sobol sequence.
        
        Parameters
        ----------
        n:
            The number of samples to generate.
        
        Returns
        -------
        rs:
            The generated samples.

        """

        S = torch.quasirandom.SobolEngine(dimension=self.approx.bases.dim)
        us = S.draw(n)
        rs = self.eval_irt(us)
        return rs

    def set_defensive(
        self, 
        tau: torch.Tensor
    ) -> None:
        """Updates the value of tau and the normalising constant 
        associated with the approximation to the target density 
        function.
        """
        
        self._tau = tau
        self._z = self.z_func + tau
        return
