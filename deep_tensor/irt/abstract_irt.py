import abc
from typing import Callable, Tuple

import torch

from ..approx_bases import ApproxBases
from ..tt_func import TTFunc
from ..constants import EPS
from ..directions import Direction
from ..input_data import InputData
from ..options import TTOptions
from ..polynomials import CDF1D
from ..tt_data import TTData


Z_MIN = torch.tensor(EPS)
Z_MAX = torch.tensor(1.0-EPS)


class AbstractIRT(abc.ABC):
    """TODO: write docstring for this."""

    def __init__(
        self, 
        potential: Callable,
        bases: ApproxBases, 
        approx: TTFunc|None,
        options: TTOptions,
        input_data: InputData,
        approx_data: TTData
    ):

        self.potential = potential 
        self.bases = bases
        self.approx = approx
        self.options = options 
        self.input_data = input_data
        self.approx_data = approx_data
        return

    @property
    @abc.abstractmethod 
    def approx(self) -> TTFunc:
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
        func: Callable,
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
        return
    
    @abc.abstractmethod
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
        return

    @abc.abstractmethod
    def build_approximation(
        self, 
        func: Callable, 
        bases: ApproxBases, 
        options: TTOptions,
        input_data: InputData,
    ) -> TTFunc:
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
    def eval_potential_local(
        self, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return

    @abc.abstractmethod
    def eval_rt_local(
        self, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return

    @abc.abstractmethod 
    def eval_rt_jac_local(
        self,
        zs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return

    @abc.abstractmethod
    def eval_irt_local_nograd(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: write docstring for this."""
        return

    @abc.abstractmethod 
    def eval_cirt_local(
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
            An n * d matrix containing samples from the approximation 
            domain.
        
        Returns
        -------
        neglogfxs:
            The negative log of the approximation to the target density 
            evaluated at each sample in xs.

        TODO: check this one with TC.

        """
        indices = self.get_transform_indices(xs.shape[1])
        ls, dldxs = self.approx.bases.approx2local(xs, indices)
        neglogfls = self.eval_potential_local(ls)
        neglogfxs = neglogfls - dldxs.log().sum(dim=1)
        return neglogfxs

    def eval_pdf(
        self, 
        xs: torch.Tensor 
    ) -> torch.Tensor: 
        """Evaluates the normalised marginal PDF at a given set of x 
        values.
        
        Parameters
        ---------
        xs: 
            An n * d matrix containing samples from the approximation 
            domain.

        Returns
        -------
        fxs:
            An n-dimensional vector containing the value of the 
            approximation to the target PDF evaluated at each element 
            in xs.
        
        """
        neglogfxs = self.eval_potential(xs)
        fxs = torch.exp(-neglogfxs)
        return fxs
    
    def eval_rt(self, xs: torch.Tensor) -> torch.Tensor:
        """Evaluates the Rosenblatt transport Z = R(X), where Z is a 
        (standard) uniform random variable and X is the target random 
        variable.

        Parameters
        ----------
        xs: 
            An n * d matrix containing samples from the approximation 
            domain.
        
        Returns
        -------
        zs:
            An n * d matrix containing the samples after applying the 
            Rosenblatt transport.
        
        """

        dim_x = xs.shape[1]
        indices = self.get_transform_indices(dim_x)
        
        ls = self.approx.bases.approx2local(xs, indices)[0]
        zs = self.eval_rt_local(ls)
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
        neglogfxs: 
            An n-dimensional vector containing the potential function
            associated with the target density evaluated at each sample
            in xs.
        
        """

        zs = torch.clamp(zs, Z_MIN, Z_MAX)
        indices = self.get_transform_indices(zs.shape[1])

        ls, neglogfls = self.eval_irt_local_nograd(zs)
        xs, dxdls = self.approx.bases.local2approx(ls, indices)
        neglogfxs = neglogfls + dxdls.log().sum(dim=1)

        return xs, neglogfxs
    
    def eval_cirt(
        self, 
        xs: torch.Tensor,
        zs: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the inverse of the conditional squared Rosenblatt 
        transport Y|X = R^{-1}(Z, X), where X is given, (X, Y) jointly 
        follow the SIRT approximation of the target distribution, and 
        Z is uniform.
        
        Parameters
        ----------
        xs:
            An n * d matrix containing samples from the approximation 
            domain.
        zs:
            An n * (m-d) matrix containing samples from [0, 1]^{m-d},
            where m is the the dimension of the joint distribution of 
            X and Y.
        
        Returns
        -------
        ys:
            An n * (m-d) matrix containing the realisations of Y 
            corresponding to the values of zs after applying the 
            (conditional) inverse Rosenblatt transport.
        neglogfys:
            An n-dimensional vector containing the potential function 
            of the approximation to the conditional density of Y|X 
            evaluated at each value of ys.
    
        """
        
        num_z, dim_z = zs.shape
        num_x, dim_x = xs.shape

        if dim_z == 0 or dim_x == 0:
            msg = "The dimensions of both X and Z should be at least 1."
            raise Exception(msg)
        
        if dim_z + dim_x != self.approx.dim:
            msg = ("The dimensions of X and Z should sum " 
                   + "to the dimension of the approximation.")
            raise Exception(msg)
        
        if num_z != num_x: 
            if num_x != 1:
                msg = "The number of samples of X and Z must be equal."
                raise Exception(msg)
            xs = xs.repeat(num_z, 1)

        if self.int_dir == Direction.FORWARD:
            inds_x = torch.arange(dim_x)
            inds_z = torch.arange(dim_x, self.approx.dim)
        elif self.int_dir == Direction.BACKWARD:
            inds_x = torch.arange(dim_z, self.approx.dim)
            inds_z = torch.arange(dim_z)
        else:
            raise NotImplementedError()
        
        ls_x = self.approx.bases.approx2local(xs, inds_x)[0]
        ls_y, neglogfys = self.eval_cirt_local(ls_x, zs)
        ys, dydlys = self.approx.bases.local2approx(ls_y, inds_z)
        neglogfys += dydlys.log().sum(dim=1)

        return ys, neglogfys
    
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
        rs = self.eval_irt_nograd(us)
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
        rs = self.eval_irt_nograd(us)
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
