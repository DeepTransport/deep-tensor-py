import abc
from copy import deepcopy
import time
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor
from torch.autograd.functional import jacobian

from .sirt import AbstractSIRT, SIRT, SavedSIRT, SUBSET2DIRECTION
from ..bridging_densities import (
    Bridge, Tempering, 
    BRIDGE2NAME, NAME2SAVEBRIDGE
)
from ..domains import Domain
from ..ftt import ApproxBases, Direction, InputData
from ..options import DIRTOptions, TTOptions
from ..polynomials import (
    Basis1D, Lagrange1, LagrangeP, Spectral, 
    POLY2NAME, NAME2POLY
)
from ..preconditioners import Preconditioner
from ..references import Reference
from ..tools.printing import dirt_info
from ..tools.saving import dict_to_h5, h5_to_dict
from ..tools import compute_f_divergence

import h5py


class AbstractDIRT(abc.ABC):

    @property 
    def preconditioner(self) -> Preconditioner:
        return self._preconditioner
    
    @preconditioner.setter
    def preconditioner(self, value: Preconditioner) -> None:
        self._preconditioner = value 
        return
    
    @property 
    def bridge(self) -> Bridge:
        return self._bridge
    
    @bridge.setter
    def bridge(self, value: Bridge) -> None:
        self._bridge = value 
        return

    @property 
    def bases(self) -> ApproxBases:
        return self._bases 
    
    @bases.setter 
    def bases(self, value: ApproxBases) -> None:
        self._bases = value 
        return
    
    @property 
    def tt_options(self) -> TTOptions:
        return self._tt_options
    
    @tt_options.setter 
    def tt_options(self, value: TTOptions) -> None:
        self._tt_options = value 
        return
    
    @property 
    def dirt_options(self) -> DIRTOptions:
        return self._dirt_options
    
    @dirt_options.setter 
    def dirt_options(self, value: DIRTOptions) -> None:
        self._dirt_options = value 
        return
    
    @property 
    def n_layers(self) -> int:
        return self.bridge.n_layers
    
    @n_layers.setter
    def n_layers(self, value: int) -> None:
        self.bridge.n_layers = value 
        return

    @property
    def sirts(self) -> Dict[int, AbstractSIRT]:
        return self._sirts
    
    @sirts.setter 
    def sirts(self, value: Dict[int, AbstractSIRT]) -> None:
        self._sirts = value 
        return
    
    @property
    def dim(self) -> int:
        return self.preconditioner.dim

    @property 
    def reference(self) -> Reference:
        return self.preconditioner.reference

    @property
    def domain(self) -> Domain:
        return self.reference.domain

    def _eval_rt_reference(
        self,
        us: Tensor,
        subset: str,
        n_layers: int
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the deep Rosenblatt transport for the pullback of 
        the target density under the preconditioning map.
        """
        
        rs = us.clone()
        neglogfus = torch.zeros(rs.shape[0])

        for i in range(n_layers):
            zs = self.sirts[i]._eval_rt(rs, subset)
            neglogsirts = self.sirts[i]._eval_potential(rs, subset)
            rs = self.reference.invert_cdf(zs)
            neglogrefs = self.reference.eval_potential(rs)[0]
            neglogfus += neglogsirts - neglogrefs

        neglogrefs = self.reference.eval_potential(rs)[0]
        neglogfus += neglogrefs

        return rs, neglogfus
    
    def _eval_irt_reference(
        self, 
        rs: Tensor, 
        subset: str = "first",
        n_layers: int | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Evaluates the deep inverse Rosenblatt transport for the 
        pullback of the target density under the preconditioning map.
        """

        if n_layers is None:
            n_layers = self.n_layers
        
        us = rs.clone()
        neglogfus = self.reference.eval_potential(us)[0]

        for i in range(n_layers-1, -1, -1):
            neglogrefs = self.reference.eval_potential(us)[0]
            zs = self.reference.eval_cdf(us)[0]
            us, neglogsirts = self.sirts[i]._eval_irt(zs, subset)
            neglogfus += neglogsirts - neglogrefs

        return us, neglogfus

    def _parse_subset(self, subset: str | None) -> str:
        
        if subset is None:
            subset = "first"
        
        subset = subset.lower()

        if subset == "last" and self.n_layers > 1:
            msg = ("When using a DIRT object with more than one layer, "
                   + "it is not possible to sample from the marginal " 
                   + "densities in the final k variables (where k < d) "
                   + "or the density of the first (d-k) variables "
                   + "conditioned on the final k variables. "
                   + "Please reverse the variable ordering or construct "
                   + "a DIRT object with a single layer.")
            raise Exception(msg)
        if subset not in ("first", "last"):
            msg = ("Invalid subset parameter encountered "
                   + f"(subset='{subset}'). Valid choices are "
                   + "'first', 'last'.")
            raise ValueError(msg)
        
        return subset

    def eval_rt(
        self,
        xs: Tensor,
        subset: str | None = None,
        n_layers: int | None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the deep Rosenblatt transport.
        
        Parameters
        ----------
        xs:
            An $n \times k$ matrix of samples from the approximation 
            domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        n_layers:
            The number of layers of the deep inverse Rosenblatt 
            transport to push the samples forward under. If not 
            specified, the samples will be pushed forward through all 
            the layers.

        Returns
        -------
        rs:
            An $n \times k$ matrix containing the composition of 
            mappings evaluated at each value of `xs`.
        neglogfxs:
            An $n$-dimensional vector containing the potential function 
            of the pullback of the reference density under the current 
            composition of mappings evaluated at each sample in `xs`.

        """
        if n_layers is None:
            n_layers = self.n_layers
        subset = self._parse_subset(subset)
        neglogdet_xs = self.preconditioner.neglogdet_Q_inv(xs, subset)
        us = self.preconditioner.Q_inv(xs, subset)
        rs, neglogfus = self._eval_rt_reference(us, subset, n_layers)
        neglogfxs = neglogfus + neglogdet_xs
        return rs, neglogfxs

    def eval_irt(
        self, 
        rs: Tensor, 
        subset: str | None = None,
        n_layers: int | None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the deep inverse Rosenblatt transport.

        Parameters
        ----------
        rs:
            An $n \times k$ matrix containing samples from the 
            reference domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        n_layers: 
            The number of layers of the deep inverse Rosenblatt 
            transport to pull the samples back under. If not specified,
            the samples will be pulled back through all the layers.

        Returns
        -------
        xs:
            An $n \times k$ matrix containing the corresponding samples 
            from the approximation domain, after applying the deep 
            inverse Rosenblatt transport.
        neglogfxs:
            An $n$-dimensional vector containing the potential function
            of the pullback of the reference density under the current 
            composition of mappings, evaluated at each sample in `xs`.

        """
        if n_layers is None:
            n_layers = self.n_layers
        subset = self._parse_subset(subset)
        us, neglogfus = self._eval_irt_reference(rs, subset, n_layers)
        xs = self.preconditioner.Q(us, subset)
        neglogdet_xs = self.preconditioner.neglogdet_Q_inv(xs, subset)
        neglogfxs = neglogfus + neglogdet_xs
        return xs, neglogfxs
    
    def eval_cirt(
        self, 
        ys: Tensor, 
        rs: Tensor, 
        subset: str = "first",
        n_layers: int | None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the conditional inverse Rosenblatt transport.

        Returns the conditional inverse Rosenblatt transport evaluated
        at a set of samples in the approximation domain. 
        
        Parameters
        ----------
        ys:
            A matrix containing samples from the approximation domain.
            The matrix should have dimensions $1 \times k$ (if the same 
            realisation of $Y$ is to be used for all samples in `rs`) 
            or $n \times k$ (if a different realisation of $Y$ is to be 
            used for each samples in `rs`).
        rs:
            An $n \times (d-k)$ matrix containing samples from the 
            reference domain.
        subset: 
            Whether `ys` corresponds to the first $k$ variables 
            (`subset='first'`) of the approximation, or the last $k$ 
            variables (`subset='last'`).
        n_layers:
            The number of layers of the DIRT object to use when 
            evaluating the CIRT. If not specified, all layers will be 
            used.
        
        Returns
        -------
        xs:
            An $n \times (d-k)$ matrix containing the realisations of 
            $X$ corresponding to the values of `rs` after applying the 
            conditional inverse Rosenblatt transport.
        neglogfxs:
            An $n$-dimensional vector containing the potential function 
            of the approximation to the conditional density of 
            $X \textbar Y$ evaluated at each sample in `xs`.
    
        """
        
        ys = torch.atleast_2d(ys)
        rs = torch.atleast_2d(rs)

        n_rs, d_rs = rs.shape
        n_ys, d_ys = ys.shape

        if d_rs == 0 or d_ys == 0:
            msg = "The dimensions of both 'ys' and 'rs' must be at least 1."
            raise ValueError(msg)
        
        if d_rs + d_ys != self.dim:
            msg = ("The dimensions of 'ys' and 'rs' must sum " 
                   + "to the dimension of the approximation.")
            raise ValueError(msg)

        if n_rs != n_ys: 
            if n_ys != 1:
                msg = ("The number of samples in 'ys' and 'rs' "
                       + "(i.e., the number of rows) must be equal.")
                raise ValueError(msg)
            ys = ys.repeat(n_rs, 1)
        
        subset = self._parse_subset(subset)
        direction = SUBSET2DIRECTION[subset]
        if direction == Direction.FORWARD:
            inds_y = torch.arange(d_ys)
            inds_x = torch.arange(d_ys, self.dim)
        elif direction == Direction.BACKWARD:
            inds_y = torch.arange(d_rs, self.dim)
            inds_x = torch.arange(d_rs)
        
        # Evaluate marginal RT
        rs_y, neglogfys = self.eval_rt(ys, subset, n_layers)

        # Evaluate joint RT
        rs_yx = torch.empty((n_rs, self.dim))
        rs_yx[:, inds_y] = rs_y 
        rs_yx[:, inds_x] = rs
        yxs, neglogfyxs = self.eval_irt(rs_yx, subset, n_layers)
        
        xs = yxs[:, inds_x]
        neglogfxs = neglogfyxs - neglogfys
        return xs, neglogfxs

    def eval_irt_pullback(
        self,
        potential: Callable[[Tensor], Tensor],
        rs: Tensor, 
        subset: str | None = None,
        n_layers: int | None = None
    ) -> Tensor:
        r"""Evaluates the pullback of a density function under the DIRT mapping.

        This function evaluates $\mathcal{T}^{\sharp}f(r)$, where 
        $\mathcal{T}(\cdot)$ denotes the inverse Rosenblatt transport and 
        $f(\cdot)$ denotes an arbitrary density function.

        Parameters
        ----------
        potential:
            A function that takes an $n \times k$ matrix of samples 
            from the approximation domain, and returns an 
            $n$-dimensional vector containing the potential function 
            associated with $f(\cdot)$ evaluated at each sample.
        rs:
            An $n \times k$ matrix containing a set of samples from the 
            reference domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        n_layers: 
            The number of layers of the deep inverse Rosenblatt 
            transport to pull the samples back under. If not specified,
            the samples will be pulled back through all the layers.

        Returns
        -------
        neglogTfrs:
            An $n$-dimensional vector containing the potential of the 
            pullback function evaluated at each element in `rs`.
        
        """
        neglogrefs = self.reference.eval_potential(rs)[0]
        xs, neglogfxs_dirt = self.eval_irt(rs, subset, n_layers)
        neglogfxs = potential(xs)
        neglogTfrs = neglogfxs + neglogrefs - neglogfxs_dirt
        return neglogTfrs
    
    def eval_cirt_pullback(
        self, 
        potential: Callable[[Tensor], Tensor],
        ys: Tensor,
        rs: Tensor,
        subset: str = "first",
        n_layers: int | None = None
    ) -> Tensor:
        r"""TODO: write docstring.
        
        Potential needs to return $f(x|y)$ (adjust notation).
        
        """
        neglogrefs = self.reference.eval_potential(rs)[0]
        ys, neglogfys_cirt = self.eval_cirt(ys, rs, subset, n_layers)
        neglogfys = potential(ys)
        neglogTfrs = neglogfys + neglogrefs - neglogfys_cirt
        return neglogTfrs

    def eval_potential(
        self, 
        xs: Tensor,
        subset: str | None = None,
        n_layers: int | None = None
    ) -> Tensor:
        r"""Evaluates the potential function.
        
        Returns the joint potential function, or the marginal potential 
        function for the first $k$ variables or the last $k$ variables,
        corresponding to the pullback of the reference measure under a 
        given number of layers of the DIRT.
        
        Parameters
        ----------
        xs:
            An $n \times k$ matrix containing a set of samples from the 
            approximation domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        n_layers:
            The number of layers of the current DIRT construction to
            use when computing the potential. If not specified, all 
            layers will be used when computing the potential.

        Returns
        -------
        neglogfxs:
            An $n$-dimensional vector containing the potential function
            of the target density evaluated at each element in `xs`.

        """
        neglogfxs = self.eval_rt(xs, subset, n_layers)[1]
        return neglogfxs
    
    def eval_pdf(
        self, 
        xs: Tensor,
        subset: str | None = None,
        n_layers: int | None = None
    ) -> Tensor: 
        r"""Evaluates the density function.
        
        Returns the joint density function, or the marginal density 
        function for the first $k$ variables or the last $k$ variables,
        corresponding to the pullback of the reference measure under 
        a given number of layers of the DIRT.
        
        Parameters
        ----------
        xs:
            An $n \times k$ matrix containing a set of samples drawn 
            from the DIRT approximation to the target density.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        n_layers:
            The number of layers of the current DIRT construction to 
            use. If not specified, all 

        Returns
        -------
        fxs:
            An $n$-dimensional vector containing the value of the 
            approximation to the target density evaluated at each 
            element in `xs`.
        
        """
        neglogfxs = self.eval_potential(xs, subset, n_layers)
        fxs = torch.exp(-neglogfxs)
        return fxs

    def eval_potential_cond(
        self, 
        ys: Tensor, 
        xs: Tensor, 
        subset: str = "first",
        n_layers: int | None = None
    ) -> Tensor:
        r"""Evaluates the conditional potential function.

        Returns the conditional potential function evaluated
        at a set of samples in the approximation domain. 
        
        Parameters
        ----------
        ys:
            An $n \times k$ matrix containing samples from the 
            approximation domain.
        xs:
            An $n \times (d-k)$ matrix containing samples from the 
            approximation domain.
        subset: 
            Whether `ys` corresponds to the first $k$ variables 
            (`subset='first'`) of the approximation, or the last $k$ 
            variables (`subset='last'`).
        n_layers:
            The number of layers of the deep inverse Rosenblatt 
            transport to push the samples forward under. If not 
            specified, the samples will be pushed forward through all 
            the layers.
        
        Returns
        -------
        neglogfxs:
            An $n$-dimensional vector containing the potential function 
            of the approximation to the conditional density of 
            $X \textbar Y$ evaluated at each sample in `xs`.
    
        """
        
        ys = torch.atleast_2d(ys)
        xs = torch.atleast_2d(xs)

        n_xs, d_xs = xs.shape
        n_ys, d_ys = ys.shape

        if d_xs == 0 or d_ys == 0:
            msg = "The dimensions of both 'ys' and 'xs' must be at least 1."
            raise ValueError(msg)
        
        if d_xs + d_ys != self.dim:
            msg = ("The dimensions of 'ys' and 'xs' must sum " 
                   + "to the dimension of the approximation.")
            raise ValueError(msg)

        if n_xs != n_ys: 
            if n_ys != 1:
                msg = ("The number of samples in 'ys' and 'xs' "
                       + "(i.e., the number of rows) must be equal.")
                raise ValueError(msg)
            ys = ys.repeat(n_xs, 1)
        
        subset = self._parse_subset(subset)

        direction = SUBSET2DIRECTION[subset]
        if direction == Direction.FORWARD:
            yxs = torch.hstack((ys, xs))
        elif direction == Direction.BACKWARD:
            yxs = torch.hstack((xs, ys))
        
        # Evaluate marginal RT
        neglogfys = self.eval_potential(ys, subset, n_layers)
        neglogfyxs = self.eval_potential(yxs, subset, n_layers)

        neglogfxs = neglogfyxs - neglogfys
        return neglogfxs

    def eval_rt_jac(
        self, 
        xs: Tensor, 
        subset: str | None = None,
        n_layers: int | None = None 
    ) -> Tensor:
        r"""Evaluates the Jacobian of the deep Rosenblatt transport.

        Evaluates the Jacobian of the mapping $R = \mathcal{R}(X)$, 
        where $R$ denotes the reference random variable, $X$ denotes 
        the approximation to the target random variable, and 
        $\mathcal{R}$ denotes the Rosenblatt transport.

        Note that element $J_{ij}$ of the Jacobian is given by
        $$J_{ij} = \frac{\partial r_{i}}{\partial x_{j}}.$$

        Parameters
        ----------
        xs:
            An $n \times d$ matrix containing a set of samples from the 
            approximation domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        n_layers: 
            The number of layers of the deep Rosenblatt transport to 
            evaluate the Jacobian for. If not specified, the Jacobian 
            for the full Rosenblatt transport will be evaluated.

        Returns
        -------
        Js:
            A $k \times n \times k$ tensor, where element $ijk$ 
            contains element $ik$ of the Jacobian for the $j$th sample 
            in `xs`.

        """

        n_xs, d_xs = xs.shape

        def _eval_rt(xs: Tensor) -> Tensor:
            xs = xs.reshape(n_xs, d_xs)
            return self.eval_rt(xs, subset, n_layers)[0].sum(dim=0)
        
        Js: Tensor = jacobian(_eval_rt, xs.flatten(), vectorize=True)
        return Js.reshape(d_xs, n_xs, d_xs)
    
    def eval_irt_jac(
        self, 
        rs: Tensor, 
        subset: str | None = None,
        n_layers: int | None = None 
    ) -> Tensor:
        r"""Evaluates the Jacobian of the deep inverse Rosenblatt transport.

        Evaluates the Jacobian of the mapping $X = \mathcal{T}(R)$, 
        where $R$ denotes the reference random variable, $X$ denotes 
        the approximation to the target random variable, and 
        $\mathcal{T}$ denotes the inverse Rosenblatt transport.

        Note that element $J_{ij}$ of the Jacobian is given by
        $$J_{ij} = \frac{\partial x_{i}}{\partial r_{j}}.$$

        Parameters
        ----------
        rs:
            An $n \times d$ matrix containing a set of samples from the 
            reference domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        n_layers: 
            The number of layers of the deep inverse Rosenblatt 
            transport to evaluate the Jacobian for. If not specified,
            the Jacobian for the full inverse Rosenblatt transport will
            be evaluated.

        Returns
        -------
        Js:
            A $k \times n \times k$ tensor, where element $ijk$ 
            contains element $ik$ of the Jacobian for the $j$th sample 
            in `rs`.

        """

        n_rs, d_rs = rs.shape

        def _eval_irt(rs: Tensor) -> Tensor:
            rs = rs.reshape(n_rs, d_rs)
            return self.eval_irt(rs, subset, n_layers)[0].sum(dim=0)
        
        Js: Tensor = jacobian(_eval_irt, rs.flatten(), vectorize=True)
        return Js.reshape(d_rs, n_rs, d_rs)

    def random(self, n: int) -> Tensor: 
        r"""Generates a set of random samples. 

        The samples are distributed according to the DIRT approximation 
        to the target density.
        
        Parameters
        ----------
        n:  
            The number of samples to generate.

        Returns
        -------
        xs:
            An $n \times d$ matrix containing the generated samples.
        
        """
        rs = self.reference.random(self.dim, n)
        xs = self.eval_irt(rs)[0]
        return xs
    
    def sobol(self, n: int) -> Tensor:
        r"""Generates a set of QMC samples.

        The samples are distributed according to the DIRT approximation 
        to the target density.
        
        Parameters
        ----------
        n:
            The number of samples to generate.
        
        Returns
        -------
        xs:
            An $n \times d$ matrix containing the generated samples.

        """
        rs = self.reference.sobol(self.dim, n)
        xs = self.eval_irt(rs)[0]
        return xs
    
    @staticmethod
    def parse_filename(fname: str) -> str:
        return fname.split(".")[0] + ".h5"

    def save(self, fname: str) -> None:
        """Saves the data associated with a `DIRT` object to a file.
        
        Parameters
        ----------
        fname:
            The name of the file to save the data associated with the 
            `DIRT` object to.
        
        """

        fname = DIRT.parse_filename(fname)
        
        d = {
            "n_layers": self.n_layers,
            "bridge": {
                "name": BRIDGE2NAME[type(self.bridge)],
                "kwargs": self.bridge.params_dict
            },
            "polys": {},
            "tt_options": self.tt_options.__dict__,
            "dirt_options": self.dirt_options.__dict__
        }
        
        for k in range(self.dim):
            poly_k = self.bases.polys[k]
            if isinstance(poly_k, Lagrange1):
                kwargs = {"num_elems": poly_k.num_elems}
            elif isinstance(poly_k, LagrangeP):
                kwargs = {
                    "num_elems": poly_k.num_elems, 
                    "order": poly_k.order
                }
            elif isinstance(poly_k, Spectral):
                kwargs = {"order": poly_k.order}
            else:
                msg = f"Unknown polynomial type: {type(poly_k)}."
                raise Exception(msg)
            d["polys"][k] = {
                "type": POLY2NAME[type(poly_k)],
                "kwargs": kwargs
            }

        for i in range(self.n_layers):
            
            d[i] = {}
            sirt = self.sirts[i]
            ftt = sirt.approx

            # Extract SIRT data
            d[i]["Bs_f"] = sirt.Bs_f
            d[i]["Bs_b"] = sirt.Bs_b
            d[i]["Rs_f"] = sirt.Rs_f
            d[i]["Rs_b"] = sirt.Rs_b
            d[i]["defensive"] = sirt.defensive
            # Extract FTT data
            d[i]["xs_samp"] = ftt.input_data.xs_samp
            d[i]["xs_debug"] = ftt.input_data.xs_debug
            d[i]["fxs_debug"] = ftt.input_data.fxs_debug
            d[i]["direction"] = ftt.tt_data.direction.value
            d[i]["cores"] = ftt.tt_data.cores

        with h5py.File(fname, "w") as f:
            dict_to_h5(f, d)


class DIRT(AbstractDIRT):
    r"""Deep (squared) inverse Rosenblatt transport.

    Parameters
    ----------
    negloglik:
        A function that receives an $n \times d$ matrix of samples and 
        returns an $n$-dimensional vector containing the negative 
        log-likelihood function evaluated at each sample.
    neglogpri:
        A function that receives an $n \times d$ matrix of samples and 
        returns an $n$-dimensional vector containing the negative 
        log-prior density evaluated at each sample.
    bases:
        A list of sets of basis functions for each dimension, or a 
        single set of basis functions (to be used in all dimensions), 
        used to construct the functional tensor trains at each 
        iteration.
    bridge: 
        An object used to generate the intermediate densities to 
        approximate at each stage of the DIRT construction.
    tt_options:
        Options for constructing the FTT approximation to the square 
        root of the ratio function (*i.e.*, the pullback of the current 
        bridging density under the existing composition of mappings) at 
        each iteration.
    dirt_options:
        Options for constructing the DIRT approximation to the 
        target density.
    prev_approx:
        A dictionary containing a set of SIRTs generated as part of 
        the construction of a previous DIRT object.

    References
    ----------
    Cui, T and Dolgov, S (2022). *[Deep composition of tensor-trains 
    using squared inverse Rosenblatt transports](https://doi.org/10.1007/s10208-021-09537-5).*
    Foundations of Computational Mathematics **22**, 1863--1922.
    
    """

    def __init__(
        self, 
        negloglik: Callable[[Tensor], Tensor],
        neglogpri: Callable[[Tensor], Tensor],
        preconditioner: Preconditioner,
        bases: Basis1D | List[Basis1D], 
        bridge: Bridge | None = None,
        tt_options: TTOptions | None = None,
        dirt_options: DIRTOptions | None = None,
        prev_approx: Dict[int, SIRT] | None = None
    ):
        
        def neglogfx_Q(us: Tensor) -> Tensor:
            xs = self.preconditioner.Q(us, "first")
            neglogdets = self.preconditioner.neglogdet_Q(us, "first")
            neglogliks = negloglik(xs)
            neglogpris = neglogpri(xs)
            self.num_eval += us.shape[0]
            return neglogpris + neglogliks + neglogdets

        if bridge is None:
            bridge = Tempering()
        if tt_options is None:
            tt_options = TTOptions()
        if dirt_options is None:
            dirt_options = DIRTOptions()

        self.neglogfx = neglogfx_Q
        self.preconditioner = preconditioner
        self.bases = ApproxBases(bases, self.domain, self.dim)
        self.bridge = bridge
        self.tt_options = tt_options
        self.dirt_options = dirt_options
        self.prev_approx = prev_approx
        self.pre_sample_size = (self.dirt_options.num_samples 
                                + self.dirt_options.num_debugs)
        self.sirts: Dict[int, SIRT] = {}
        self.num_eval = 0
        self.log_z = 0.0

        self._build()
        return

    def _get_potential_to_density(
        self, 
        neglogratios: Tensor, 
        xs: Tensor
    ) -> Tensor:
        """Returns the function we aim to approximate (i.e., the 
        square root of the ratio function divided by the weighting 
        function associated with the reference measure).

        Parameters
        ----------
        neglogratios:
            An n-dimensional vector containing the negative logarithm 
            of the ratio function associated with each sample.
        xs:
            An n * d matrix containing a set of samples distributed 
            according to the current bridging density.
        
        Returns
        -------
        ys:
            An n-dimensional vector containing evaluations of the 
            target function at each sample in xs.
        
        """
        neglogwrs = self.bases.eval_measure_potential(xs)[0]
        log_ys = -0.5 * (neglogratios - neglogwrs)
        return torch.exp(log_ys)

    def _get_inputdata(
        self,
        xs: Tensor, 
        neglogratios: Tensor 
    ) -> InputData:
        """Generates a set of input data and debugging samples used to 
        initialise DIRT.
        
        Parameters
        ----------
        xs:
            An n * d matrix containing samples distributed according to
            the current bridging density.
        neglogratios:
            A n-dimensional vector containing the negative logarithm of
            the current ratio function evaluated at each sample in xs.
        
        Returns
        -------
        input_data:
            An InputData object containing a set of samples used to 
            construct the FTT approximation to the target function, and 
            (if debugging samples are requested) a set of debugging 
            samples and the value of the target function evaluated 
            at each debugging sample.
            
        """

        if self.dirt_options.num_debugs == 0:
            return InputData(xs)
        
        indices = torch.arange(self.dirt_options.num_samples)
        indices_debug = (torch.arange(self.dirt_options.num_debugs)
                         + self.dirt_options.num_samples)

        fxs_debug = self._get_potential_to_density(
            neglogratios[indices_debug], 
            xs[indices_debug]
        )

        return InputData(xs[indices], xs[indices_debug], fxs_debug)

    def _get_new_layer(self, xs: Tensor, neglogratios: Tensor) -> SIRT:
        """Constructs a new SIRT to add to the current composition of 
        SIRTs.

        Parameters
        ----------
        xs:
            An n * d matrix containing samples distributed according to
            the current bridging density.
        neglogratios:
            An n-dimensional vector containing the negative log-ratio 
            function evaluated at each element in xs.

        Returns
        -------
        sirt:
            The squared inverse Rosenblatt transport approximation to 
            the next bridging density.
        
        """

        def updated_func(rs: Tensor) -> Tensor:

            neglogrefs_rs = self.reference.eval_potential(rs)[0]

            xs, neglogfxs_dirt = self._eval_irt_reference(rs)
            neglogrefs = self.reference.eval_potential(xs)[0]
            neglogfxs = self.neglogfx(xs)

            neglogratios = self.bridge._get_ratio_func(
                self.dirt_options.method,
                neglogrefs_rs,
                neglogrefs,
                neglogfxs,
                neglogfxs_dirt
            )

            return neglogratios

        if self.prev_approx is None:
            
            # Generate debugging and initialisation samples
            input_data = self._get_inputdata(xs, neglogratios)

            if self.n_layers == 0:
                approx = None 
                tt_data = None
            else:
                # Use previous approximation as a starting point
                approx = deepcopy(self.sirts[self.n_layers-1].approx)
                tt_data = deepcopy(self.sirts[self.n_layers-1].approx.tt_data)

            sirt = SIRT(
                updated_func,
                preconditioner=self.preconditioner,
                bases=self.bases.polys,
                prev_approx=approx,
                options=self.tt_options,
                input_data=input_data,
                tt_data=tt_data,
                defensive=self.dirt_options.defensive
            )
        
        else:
            
            ind_prev = max(self.prev_approx.keys())
            sirt_prev = self.prev_approx[min(ind_prev, self.n_layers)]
            
            input_data = self._get_inputdata(xs, neglogratios)

            sirt = SIRT(
                updated_func,
                preconditioner=self.preconditioner,
                bases=sirt_prev.approx.bases.polys,
                options=self.tt_options,
                input_data=input_data, 
                defensive=self.dirt_options.defensive
            )
        
        return sirt

    def _print_progress(
        self,
        log_weights: Tensor, 
        neglogrefs: Tensor, 
        neglogfxs: Tensor, 
        neglogfxs_dirt: Tensor,
        cum_time: float
    ) -> None:

        msg = [
            f"Iter: {self.n_layers+1:=2}",
            f"Cum. Fevals: {self.num_eval:=.2e}",
            f"Cum. Time: {cum_time:=.2e} s"
        ]

        msg_bridge = self.bridge._get_diagnostics(
            log_weights, 
            neglogrefs, 
            neglogfxs, 
            neglogfxs_dirt
        )

        dirt_info(" | ".join(msg + msg_bridge))
        return
    
    def _build(self) -> None:
        """Constructs a DIRT to approximate a given probability 
        density.
        """

        t0 = time.time()
        
        while True:

            # Draw a new set of samples from the reference, then 
            # push them forward through the current composition of 
            # (inverse) mappings
            rs = self.reference.random(self.dim, self.pre_sample_size)
            neglogrefs_rs = self.reference.eval_potential(rs)[0]

            xs, neglogfxs_dirt = self._eval_irt_reference(rs)
            neglogrefs = self.reference.eval_potential(xs)[0]
            neglogfxs = self.neglogfx(xs)
        
            self.bridge._adapt_density(
                self.dirt_options.method, 
                neglogrefs, 
                neglogfxs, 
                neglogfxs_dirt
            )

            neglogratios = self.bridge._get_ratio_func(
                self.dirt_options.method, 
                neglogrefs_rs,
                neglogrefs, 
                neglogfxs, 
                neglogfxs_dirt
            )
            
            log_weights = self.bridge._compute_log_weights(
                neglogrefs,
                neglogfxs, 
                neglogfxs_dirt
            )

            if self.dirt_options.verbose:
                cum_time = time.time() - t0
                self._print_progress(
                    log_weights, 
                    neglogrefs, 
                    neglogfxs, 
                    neglogfxs_dirt,
                    cum_time
                )

            rs, neglogratios = self.bridge._reorder(rs, neglogratios, log_weights)
            self.sirts[self.n_layers] = self._get_new_layer(rs, neglogratios)

            self.log_z += self.sirts[self.n_layers].z.log()
            self.num_eval += self.sirts[self.n_layers].approx.num_eval

            self.n_layers += 1
            if self.bridge.is_last:

                rs = self.reference.random(self.dim, self.pre_sample_size)
                xs, neglogfxs_dirt = self._eval_irt_reference(rs)
                neglogfxs = self.neglogfx(xs)
                dhell2 = compute_f_divergence(-neglogfxs_dirt, -neglogfxs)
                t1 = time.time()

                if self.dirt_options.verbose:
                    
                    dirt_info("DIRT construction complete.")
                    dirt_info(f" • Layers: {self.n_layers}.")
                    dirt_info(f" • Total function evaluations: {self.num_eval:,}.")
                    dirt_info(f" • Total time: {t1-t0:,.2} s.")
                    dirt_info(f" • DHell: {dhell2.sqrt():.4f}.")

                return


class SavedDIRT(AbstractDIRT):
    r"""Reconstructs a `DIRT` object from a file.

    This class has the same methods as a regular `DIRT` object.
    
    Parameters
    ----------
    fname: 
        The name of the file to read the `DIRT` object from.
    
    """

    def __init__(self, fname: str, preconditioner: Preconditioner):

        fname = DIRT.parse_filename(fname)

        with h5py.File(fname, "r") as f:
            d = h5_to_dict(f)

        bridge_name = d["bridge"]["name"]
        bridge_kwargs = d["bridge"]["kwargs"]
        
        self.preconditioner = preconditioner
        self.bridge = NAME2SAVEBRIDGE[bridge_name](**bridge_kwargs)
        self.polys = self._parse_polynomials(d["polys"])
        self.bases = ApproxBases(self.polys, self.domain, self.dim)
        self.tt_options = TTOptions(**d["tt_options"])
        self.dirt_options = DIRTOptions(**d["dirt_options"])
        self.sirts = {
            i: SavedSIRT(
                d[str(i)],
                self.preconditioner, 
                self.polys,
                self.tt_options
            ) for i in range(d["n_layers"])
        }

        return
    
    def _parse_polynomials(self, d: Dict) -> List[Basis1D]:
        """Extracts a set of saved polynomial bases for each dimension."""
        polys = []
        for j in range(self.dim):
            poly_name = d[str(j)]["type"]
            poly_kwargs = d[str(j)]["kwargs"]
            polys.append(NAME2POLY[poly_name](**poly_kwargs))
        return polys