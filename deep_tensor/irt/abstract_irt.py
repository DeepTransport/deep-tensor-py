import abc
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.autograd.functional import jacobian
from torch.quasirandom import SobolEngine

from ..constants import EPS
from ..ftt import ApproxBases, Direction, InputData, TTData, TTFunc
from ..options import TTOptions
from ..polynomials import CDF1D


Z_MIN = torch.tensor(EPS)
Z_MAX = torch.tensor(1.0-EPS)

PotentialFunc = Callable[[Tensor], Tensor]


class AbstractIRT(abc.ABC):

    def __init__(
        self, 
        potential: Callable[[Tensor], Tensor],
        bases: ApproxBases|None, 
        prev_approx: TTFunc|None,
        options: TTOptions|None,
        input_data: InputData|None,
        tt_data: TTData|None
    ):

        # TODO: allow for dimension to be passed in
        if bases is None and prev_approx is None:
            msg = ("Must pass in a previous approximation or a set of "
                   + "approximation bases.")
            raise Exception(msg)

        if prev_approx is not None:
            bases = prev_approx.bases 
            options = prev_approx.options
            tt_data = prev_approx.tt_data

        if options is None:
            options = TTOptions()
        
        if input_data is None:
            input_data = InputData()

        self.potential = potential 
        self.bases = bases
        self.dim = self.bases.dim
        # self.approx = prev_approx
        self.options = options 
        self.input_data = input_data
        self.tt_data = tt_data
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

    @abc.abstractmethod
    def _potential2density(
        self, 
        potential_func: Callable[[Tensor], Tensor], 
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
    def _eval_potential_local(self, ls: Tensor, direction: Direction) -> Tensor:
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
    def _eval_rt_local(self, ls: Tensor, direction: Direction) -> Tensor:
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
    def _eval_irt_local(
        self, 
        zs: Tensor, 
        direction: Direction
    ) -> Tuple[Tensor, Tensor]:
        """Converts a set of realisations of a standard uniform 
        random variable, Z, to the corresponding realisations of the 
        local target random variable, by applying the inverse 
        Rosenblatt transport.
        
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
    def _eval_cirt_local(
        self, 
        ls_x: Tensor, 
        zs: Tensor,
        direction: Direction
    ) -> Tensor:
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
    def _eval_potential_grad_local(
        self, 
        ls: Tensor, 
        direction: Direction
    ) -> Tensor:
        """Evaluates the gradient of the potential function.
        
        Parameters
        ----------
        ls:
            An n * d set of samples from the local domain.
        
        Returns 
        -------
        grads:
            An n * d matrix containing the gradient of the potential 
            function at each element in ls.
        
        """
        return

    @abc.abstractmethod 
    def _eval_rt_jac_local(self, zs: Tensor, direction: Direction) -> Tensor:
        """Evaluates the Jacobian of the Rosenblatt transport.
        
        Parameters
        ----------
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
    
    @staticmethod
    def _get_direction(subset: str|None) -> Direction:
        """Converts the subset parameter into the direction 
        corresponding to the marginalisation tensors that should be 
        used.
        """

        if subset is None:
            return Direction.FORWARD
        
        subset = subset.lower()
        if subset == "first":
            return Direction.FORWARD
        elif subset == "last":
            return Direction.BACKWARD
        
        msg = ("Unknown value of 'subset' found. Acceptable values "
               + "are 'first', 'last', or None.")
        raise Exception(msg)

    def _get_transform_indices(self, dim_z: int, direction: Direction) -> Tensor:
        """TODO: write docstring."""

        if direction == Direction.FORWARD:
            return torch.arange(dim_z)
        elif direction == Direction.BACKWARD:
            return torch.arange(self.dim-dim_z, self.dim)

    def _eval_potential_grad_autodiff(self, xs: Tensor, subset: str) -> Tensor:
        """Evaluates the gradient of the potential using autodiff."""

        xs_shape = xs.shape

        def _eval_potential(xs: Tensor) -> Tensor:
            xs = xs.reshape(*xs_shape)
            return self.eval_potential(xs, subset).sum(dim=0)
        
        derivs = jacobian(_eval_potential, xs.flatten(), vectorize=True)
        return derivs.reshape(*xs_shape)

    def _eval_rt_jac_autodiff(self, xs: Tensor, subset: str) -> Tensor:
        """Evaluates the gradient of the Rosenblatt transport using 
        autodiff.
        """

        n_xs, d_xs = xs.shape

        def _eval_rt(xs: Tensor) -> Tensor:
            xs = xs.reshape(n_xs, d_xs)
            return self.eval_rt(xs, subset).sum(dim=0)
        
        Js = jacobian(_eval_rt, xs.flatten(), vectorize=True)
        return Js.reshape(d_xs, n_xs, d_xs)

    def set_tau(self, tau: float|Tensor) -> None:
        r"""Updates the defensive parameter, $\tau$.
        
        Parameters
        ----------
        tau: 
            The updated value for $\tau$, the defensive parameter of 
            the IRT.
        
        """
        self._tau = tau
        self._z = self.z_func + tau
        return
    
    def eval_potential(
        self, 
        xs: Tensor, 
        subset: str|None = None
    ) -> Tensor:
        r"""Evaluates the potential function.

        Returns the joint potential function, or the marginal potential 
        function for the first $k$ variables or the last $k$ variables,
        evaluated at a set of samples.

        Parameters
        ----------
        xs:
            An $n \times k$ matrix (where $1 \leq k \leq d$) containing 
            samples from the approximation domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        
        Returns
        -------
        neglogfxs:
            The potential function of the approximation to the target 
            density evaluated at each sample in `xs`.

        """
        direction = AbstractIRT._get_direction(subset)
        indices = self._get_transform_indices(xs.shape[1], direction)
        ls, dldxs = self.bases.approx2local(xs, indices)
        neglogfls = self._eval_potential_local(ls, direction)
        neglogfxs = neglogfls - dldxs.log().sum(dim=1)
        return neglogfxs

    def eval_pdf(
        self, 
        xs: Tensor,
        subset: str|None = None
    ) -> Tensor: 
        r"""Evaluates the density function.

        Returns the joint density function, or the marginal density 
        function for the first $k$ variables or the last $k$ variables, 
        evaluated at a set of samples.
        
        Parameters
        ----------
        xs:
            An $n \times k$ matrix (where $1 \leq k \leq d$) containing 
            samples from the approximation domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        fxs:
            An $n$-dimensional vector containing the value of the 
            approximation to the target density evaluated at each 
            element in `xs`.
        
        """
        neglogfxs = self.eval_potential(xs, subset)
        fxs = torch.exp(-neglogfxs)
        return fxs
    
    def eval_rt(
        self, 
        xs: Tensor,
        subset: str|None = None
    ) -> Tensor:
        r"""Evaluates the Rosenblatt transport.

        Returns the joint Rosenblatt transport, or the marginal 
        Rosenblatt transport for the first $k$ variables or the last 
        $k$ variables, evaluated at a set of samples.

        Parameters
        ----------
        xs: 
            An $n \times k$ matrix (where $1 \leq k \leq d$) containing 
            samples from the approximation domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        
        Returns
        -------
        zs:
            An $n \times k$ matrix containing the corresponding 
            samples, from the unit hypercube, after applying the 
            Rosenblatt transport.

        """
        direction = AbstractIRT._get_direction(subset)
        indices = self._get_transform_indices(xs.shape[1], direction)
        ls = self.approx.bases.approx2local(xs, indices)[0]
        zs = self._eval_rt_local(ls, direction)
        return zs
    
    def eval_irt(
        self, 
        zs: Tensor,
        subset: str|None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the inverse Rosenblatt transport.
        
        Returns the joint inverse Rosenblatt transport, or the marginal 
        inverse Rosenblatt transport for the first $k$ variables or the 
        last $k$ variables, evaluated at a set of samples.
        
        Parameters
        ----------
        zs: 
            An $n \times k$ matrix containing samples from the unit 
            hypercube.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        
        Returns
        -------
        xs: 
            An $n \times k$ matrix containing the corresponding samples 
            from the approximation to the target density function.
        neglogfxs: 
            An $n$-dimensional vector containing the approximation to 
            the potential function evaluated at each sample in `xs`.
        
        """
        zs = torch.clamp(zs, Z_MIN, Z_MAX)
        direction = AbstractIRT._get_direction(subset)
        indices = self._get_transform_indices(zs.shape[1], direction)
        ls, neglogfls = self._eval_irt_local(zs, direction)
        xs, dxdls = self.bases.local2approx(ls, indices)
        neglogfxs = neglogfls + dxdls.log().sum(dim=1)
        return xs, neglogfxs
    
    def eval_cirt(
        self, 
        xs: Tensor, 
        zs: Tensor, 
        subset: str|None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the conditional inverse Rosenblatt transport.

        Returns the conditional inverse Rosenblatt transport evaluated
        at a set of samples in the approximation domain. 
        
        The conditional inverse Rosenblatt transport takes the form
        $$Y|X = R^{-1}(R_{k}(X), Z),$$
        where $X$ is a $k$-dimensional random variable, $Z$ is an 
        $n-k$-dimensional uniform random variable, $R(\,\cdot\,)$ 
        denotes the (full) Rosenblatt transport, and $R_{k}(\,\cdot\,)$ 
        denotes the Rosenblatt transport for the first (or last) $k$ 
        variables.
        
        Parameters
        ----------
        xs:
            An $n \times k$ matrix containing samples from the 
            approximation domain.
        zs:
            An $n \times (d-k)$ matrix containing samples from the unit 
            hypercube of dimension $d-k$.
        subset: 
            Whether `xs` corresponds to the first $k$ variables 
            (`subset='first'`) of the approximation, or the last $k$ 
            variables (`subset='last'`).
        
        Returns
        -------
        ys:
            An $n \times (d-k)$ matrix containing the realisations of 
            $Y$ corresponding to the values of `zs` after applying the 
            conditional inverse Rosenblatt transport.
        neglogfys:
            An $n$-dimensional vector containing the potential function 
            of the approximation to the conditional density of 
            $Y \textbar X$ evaluated at each sample in `ys`.
    
        """
        
        n_zs, d_zs = zs.shape
        n_xs, d_xs = xs.shape

        if d_zs == 0 or d_xs == 0:
            msg = "The dimensions of both X and Z must be at least 1."
            raise Exception(msg)
        
        if d_zs + d_xs != self.dim:
            msg = ("The dimensions of X and Z must sum " 
                   + "to the dimension of the approximation.")
            raise Exception(msg)
        
        if n_zs != n_xs: 
            if n_xs != 1:
                msg = "The number of samples of X and Z must be equal."
                raise Exception(msg)
            xs = xs.repeat(n_zs, 1)
        
        direction = AbstractIRT._get_direction(subset)
        if direction == Direction.FORWARD:
            inds_x = torch.arange(d_xs)
            inds_z = torch.arange(d_xs, self.dim)
        elif direction == Direction.BACKWARD:
            inds_x = torch.arange(d_zs, self.dim)
            inds_z = torch.arange(d_zs)
        
        ls_x = self.bases.approx2local(xs, inds_x)[0]
        ls_y, neglogfys = self._eval_cirt_local(ls_x, zs, direction)
        ys, dydlys = self.bases.local2approx(ls_y, inds_z)
        neglogfys += dydlys.log().sum(dim=1)

        return ys, neglogfys
    
    def eval_potential_grad(
        self, 
        xs: Tensor, 
        method: str = "autodiff",
        subset: str|None = None
    ) -> Tensor:
        r"""Evaluates the gradient of the potential function.
        
        Parameters
        ----------
        xs:
            An $n \times k$ matrix containing samples from the 
            approximation domain.
        method: 
            The method by which to compute the gradient. This can be 
            `autodiff`, or `manual`. Generally, `manual` is faster than 
            `autodiff`, but can only be used to evaluate the gradient 
            of the full potential function (*i.e.*, when $k=d$).
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        grads:
            An $n \times k$ matrix containing the gradient of the 
            potential function evaluated at each sample in `xs`.

        """

        method = method.lower()
        if method not in ("manual", "autodiff"):
            raise Exception("Unknown method.")

        if method == "autodiff":
            TTFunc._check_sample_dim(xs, self.dim)
            grad = self._eval_potential_grad_autodiff(xs, subset)
            return grad
        
        TTFunc._check_sample_dim(xs, self.dim, strict=True)
        ls, dldxs = self.bases.approx2local(xs)
        grad = self._eval_potential_grad_local(ls)
        grad *= dldxs
        return grad

    def eval_rt_jac(
        self, 
        xs: Tensor, 
        method: str = "autodiff",
        subset: str|None = None
    ) -> Tensor:
        r"""Evaluates the Jacobian of the Rosenblatt transport.

        Evaluates the Jacobian of the mapping $Z = R(X)$, where $Z$ is 
        a standard $k$-dimensional uniform random variable and $X$ is 
        the approximation to the target random variable. 

        Note that element $J_{ij}$ of the Jacobian is given by
        $$J_{ij} = \frac{\partial z_{i}}{\partial x_{j}}.$$

        Parameters
        ----------
        xs:
            An $n \times d$ matrix containing a set of samples from the 
            approximation domain.
        method:
            The method by which to compute the Jacobian. This can be 
            `autodiff`, or `manual`. Generally, `manual` is faster than 
            `autodiff`, but can only be used to evaluate the Jacobian 
            of the full Rosenblatt transport (*i.e.*, when $k=d$).
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        Jacs:
            A $k \times n \times k$ tensor, where element $ijk$ 
            contains element $ik$ of the Jacobian for the $j$th sample 
            in `xs`.

        """

        direction = AbstractIRT._get_direction(subset)
        method = method.lower()
        if method not in ("manual", "autodiff"):
            raise Exception("Unknown method.")

        if method == "autodiff":
            TTFunc._check_sample_dim(xs, self.dim)
            Jacs = self._eval_rt_jac_autodiff(xs, subset)
            return Jacs
        
        TTFunc._check_sample_dim(xs, self.dim, strict=True)
        ls, dldxs = self.bases.approx2local(xs)
        Jacs = self._eval_rt_jac_local(ls, direction)
        for k in range(self.dim):
            Jacs[:, :, k] *= dldxs[:, k]
        return Jacs
    
    # def eval_rt_jac_prod(self, xs: Tensor, vs: Tensor) -> Tensor:
    #     """
    #     xs: samples to compute RT with.
    #     vs: samples to compute J(x)v with
    #     """
    #     TTFunc._check_sample_dim(xs, self.dim, strict=True)
    #     Js = self._eval_rt_jac_prod_autodiff(xs, vs)
    #     return Js

    # def _eval_rt_jac_prod_autodiff(self, xs: Tensor, vs: Tensor) -> Tensor:

    #     n_xs, d_xs = xs.shape

    #     def _eval_rt(xs: Tensor) -> Tensor:
    #         xs = xs.reshape(n_xs, self.dim)
    #         return self.eval_rt(xs).sum(dim=0)
        
    #     Jvs: Tensor = torch.func.jvp(  # torch.func.jvp?
    #         _eval_rt, 
    #         primals=xs,#.flatten(), 
    #         tangents=vs#.flatten()
    #     )[1]

    #     print(Jvs.shape)

    #     return Jvs.reshape(n_xs, d_xs)

    def random(self, n: int) -> Tensor: 
        """Generates a set of random samples. 
        
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
        xs = self.eval_irt(zs)[0]
        return xs 
    
    def sobol(self, n: int) -> Tensor:
        """Generates a set of QMC samples.
        
        Parameters
        ----------
        n:
            The number of samples to generate.
        
        Returns
        -------
        xs:
            The generated samples.

        """
        S = SobolEngine(dimension=self.dim)
        zs = S.draw(n)
        xs = self.eval_irt(zs)[0]
        return xs