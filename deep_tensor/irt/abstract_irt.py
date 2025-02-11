import abc
from typing import Callable, Tuple

import torch
from torch import Tensor

from ..constants import EPS
from ..ftt import ApproxBases, Direction, InputData, TTData, TTFunc
from ..options import TTOptions
from ..polynomials import CDF1D


Z_MIN = torch.tensor(EPS)
Z_MAX = torch.tensor(1.0-EPS)


class AbstractIRT(abc.ABC):

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
        """The FTT approximation to the target function.
        """
        return

    @property
    @abc.abstractmethod  
    def tau(self) -> Tensor:
        """The defensive term (used to ensure that the tails of the 
        approximation are sufficiently heavy).
        """
        return 

    @property
    @abc.abstractmethod 
    def z_func(self) -> Tensor:
        """The normalising constant of the function approximation part 
        of the target density.
        """
        return 
    
    @property
    @abc.abstractmethod  
    def z(self) -> Tensor:
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
    def int_dir(self) -> Direction:
        """The direction in which to integrate over the approximation.
        """
        return

    @property
    def dim(self) -> int:
        """The dimension of the target PDF.
        """
        return self.bases.dim

    @abc.abstractmethod 
    def marginalise(self, direction: Direction=Direction.FORWARD) -> None:
        """Computes each coefficient tensor (B_k) required to evaluate 
        the marginal functions in each dimension, as well as the 
        normalising constant, z. 

        Parameters
        ----------
        direction:
            The direction in which to iterate over the tensor cores.

        Returns
        -------
        None

        Notes
        -----
        Updates self.Bs, self.z_func, self.z.

        References
        ----------
        Cui and Dolgov (2022, Sec. 3.1). Deep composition of tensor 
        trains using squared inverse Rosenblatt transports.

        """
        return

    @abc.abstractmethod
    def potential2density(
        self,
        func: Callable[[Tensor], Tensor],
        ls: Tensor
    ) -> Tensor:
        """Computes the value of the target function being approximated 
        by the FTT for a sample, or set of samples, from the local 
        domain. 

        Parameters
        ----------
        potential_func:
            A function that returns the potential of the target density 
            function at a given point in the approximation domain.
        ls:
            An n * d matrix containing a set of n samples from the 
            local domain.

        Returns
        ------
        gs:
            An n-dimensional vector containing the value of the 
            function being approximated by the FTT for each sample in 
            ls.
        
        """
        return
    
    @abc.abstractmethod
    def get_potential2density(self, ys: Tensor, zs: Tensor) -> Tensor:
        """TODO: implement."""
        return

    @abc.abstractmethod
    def build_approximation(
        self, 
        target_func: Callable[[Tensor], Tensor], 
        bases: ApproxBases, 
        options: TTOptions,
        input_data: InputData,
    ) -> TTFunc:
        """Constructs a functional approximation to a given target 
        density function.

        Parameters
        ----------
        target_func:
            A function that returns the value of the target function 
            evaluated at a set of samples from the local domain.
        bases:
            The polynomial bases associated with each dimension.
        options:
            Options used when constructing the FTT.
        input_data:
            An object containing samples used to initialise and 
            evaluate the quality of the FTT approximation to the 
            target function.

        Returns
        -------
        approx:
            The FTT approximation to the target function.

        """
        return

    @abc.abstractmethod 
    def eval_potential_local(self, ls: Tensor) -> Tensor:
        """Evaluates the normalised (marginal) PDF represented by the 
        squared FTT.
        
        Parameters
        ----------
        ls:
            An n * d matrix containing a set of samples from the local 
            domain.

        Returns
        -------
        neglogfls:
            An n-dimensional vector containing the approximation to the 
            target density function (transformed into the local domain) 
            at each element in ls.
        
        """
        return

    @abc.abstractmethod
    def eval_rt_local(self, ls: Tensor) -> Tensor:
        """Evaluates the Rosenblatt transport Z = R(L), where L is the 
        target random variable mapped into the local domain, and Z is 
        uniform.

        Parameters
        ----------
        ls:
            An n * d matrix containing samples from the local domain.
        
        Returns
        -------
        zs:
            An n * d matrix containing the result of applying the 
            inverse Rosenblatt transport to each sample in ls.
        
        """
        return

    @abc.abstractmethod
    def eval_irt_local(self, zs: Tensor) -> Tuple[Tensor, Tensor]:
        """Converts a set of realisations of a standard uniform 
        random variable, Z, to the corresponding realisations of the 
        local (i.e., defined on [-1, 1]) target random variable, by 
        applying the inverse Rosenblatt transport.
        
        Parameters
        ----------
        zs: 
            An n * d matrix containing values on [0, 1]^d.

        Returns
        -------
        ls:
            An n * d matrix containing the corresponding samples of the 
            target random variable mapped into the local domain.
        neglogfls:
            The local potential function associated with the 
            approximation to the target density, evaluated at each 
            sample.

        """
        return

    @abc.abstractmethod 
    def eval_cirt_local(self, ls_x: Tensor, zs: Tensor) -> Tensor:
        """Evaluates the inverse of the conditional squared Rosenblatt 
        transport.
        
        Parameters
        ----------
        ls_x:
            An n * m matrix containing samples from the local domain.
        zs:
            An n * (d-m) matrix containing samples from [0, 1]^{d-m},
            where m is the the dimension of the joint distribution of 
            X and Y.
        
        Returns
        -------
        ys:
            An n * (d-m) matrix containing the realisations of Y 
            corresponding to the values of zs after applying the 
            conditional inverse Rosenblatt transport.
        neglogfys:
            An n-dimensional vector containing the potential function 
            of the approximation to the conditional density of Y|X 
            evaluated at each sample in ys.
    
        """
        return

    @abc.abstractmethod 
    def eval_rt_jac_local(self, zs: Tensor) -> Tensor:
        """Evaluates the Jacobian of the Rosenblatt transport.
        
        Parameters
        ----------
        ls:
            An n * d set of samples from the local domain.
        zs: 
            An n * d matrix corresponding to evaluations of the 
            Rosenblatt transport at each sample in ls.
        
        Returns
        -------
        Js:
            A d * (d*n) matrix, where each d * d block contains the 
            Jacobian of the Rosenblatt transport evaluated at a given 
            sample: that is, J_ij = dz_i / dl_i.

        """
        return

    def get_transform_indices(self, dim_z: int) -> Tensor:
        """TODO: write docstring."""

        if self.int_dir == Direction.FORWARD:
            return torch.arange(dim_z)
        elif self.int_dir == Direction.BACKWARD:
            return torch.arange(self.dim-dim_z, self.dim)
        
        msg = "int_dir must be specified."
        raise Exception(msg)

    def eval_potential(self, xs: Tensor) -> Tensor:
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

        """
        indices = self.get_transform_indices(xs.shape[1])
        ls, dldxs = self.bases.approx2local(xs, indices)
        neglogfls = self.eval_potential_local(ls)
        neglogfxs = neglogfls - dldxs.log().sum(dim=1)
        return neglogfxs

    def eval_pdf(self, xs: Tensor) -> Tensor: 
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
    
    def eval_rt(self, xs: Tensor) -> Tensor:
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
        d_xs = xs.shape[1]
        indices = self.get_transform_indices(d_xs)
        ls = self.approx.bases.approx2local(xs, indices)[0]
        zs = self.eval_rt_local(ls)
        return zs
    
    def eval_irt(self, zs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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

        ls, neglogfls = self.eval_irt_local(zs)
        xs, dxdls = self.bases.local2approx(ls, indices)
        neglogfxs = neglogfls + dxdls.log().sum(dim=1)

        return xs, neglogfxs
    
    def eval_cirt(self, xs: Tensor, zs: Tensor ) -> Tuple[Tensor, Tensor]:
        """Evaluates the inverse of the conditional squared Rosenblatt 
        transport Y|X = R^{-1}(Z, X), where X is given, (X, Y) jointly 
        follow the SIRT approximation of the target distribution, and 
        Z is uniform.
        
        Parameters
        ----------
        xs:
            An n * m matrix containing samples from the approximation 
            domain.
        zs:
            An n * (d-m) matrix containing samples from [0, 1]^{d-m},
            where m is the the dimension of the joint distribution of 
            X and Y.
        
        Returns
        -------
        ys:
            An n * (d-m) matrix containing the realisations of Y 
            corresponding to the values of zs after applying the 
            conditional inverse Rosenblatt transport.
        neglogfys:
            An n-dimensional vector containing the potential function 
            of the approximation to the conditional density of Y|X 
            evaluated at each sample in ys.
    
        """
        
        n_zs, d_zs = zs.shape
        n_xs, d_xs = xs.shape

        if d_zs == 0 or d_xs == 0:
            msg = "The dimensions of both X and Z should be at least 1."
            raise Exception(msg)
        
        if d_zs + d_xs != self.dim:
            msg = ("The dimensions of X and Z should sum " 
                   + "to the dimension of the approximation.")
            raise Exception(msg)
        
        if n_zs != n_xs: 
            if n_xs != 1:
                msg = "The number of samples of X and Z must be equal."
                raise Exception(msg)
            xs = xs.repeat(n_zs, 1)

        if self.int_dir == Direction.FORWARD:
            inds_x = torch.arange(d_xs)
            inds_z = torch.arange(d_xs, self.dim)
        elif self.int_dir == Direction.BACKWARD:
            inds_x = torch.arange(d_zs, self.dim)
            inds_z = torch.arange(d_zs)
        
        ls_x = self.bases.approx2local(xs, inds_x)[0]
        ls_y, neglogfys = self.eval_cirt_local(ls_x, zs)
        ys, dydlys = self.bases.local2approx(ls_y, inds_z)
        neglogfys += dydlys.log().sum(dim=1)

        return ys, neglogfys
    
    def eval_rt_jac(self, xs: Tensor) -> Tensor:
        """Evaluates the Jacobian of the squared Rosenblatt transport 
        Z = R(X), where Z is the uniform random variable and X is the 
        target random variable.

        """

        TTFunc._check_sample_dim(xs, self.dim, strict=True)

        ls, dldxs = self.bases.approx2local(xs)
        Js = self.eval_rt_jac_local(ls)

        n_ls, d_ls = ls.shape
        for k in range(n_ls):
            inds = k * d_ls + torch.arange(d_ls)
            Js[:, inds] *= dldxs[k]

        return Js

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
        xs:
            The generated samples.
        
        """
        zs = torch.rand(n, self.dim)
        xs = self.eval_irt(zs)
        return xs 
    
    def sobol(self, n: int) -> torch.Tensor:
        """Generates a set of QMC samples from the approximation to the 
        target density function using a Sobol sequence.
        
        Parameters
        ----------
        n:
            The number of samples to generate.
        
        Returns
        -------
        xs:
            The generated samples.

        """
        S = torch.quasirandom.SobolEngine(dimension=self.dim)
        zs = S.draw(n)
        xs = self.eval_irt(zs)
        return xs

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
