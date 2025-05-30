import abc
from copy import deepcopy
import time
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor
from torch.autograd.functional import jacobian

from .sirt import AbstractSIRT, SIRT, SavedSIRT
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
        xs: Tensor,
        n_layers: Tensor = torch.inf,
        subset: str | None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the deep Rosenblatt transport.
        
        Parameters
        ----------
        xs:
            An $n \times k$ matrix of random variables in the reference
            domain.
        n_layers:
            The number of layers of the deep inverse Rosenblatt 
            transport to push the samples forward under. If not 
            specified, the samples will be pushed forward through all 
            the layers.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        rs:
            An $n \times k$ matrix containing the composition of 
            mappings evaluated at each value of `xs`.
        neglogfxs:
            An $n$-dimensional vector containing the potential function 
            of the pullback of the reference density under the current 
            composition of mappings, evaluated at each sample in `xs`.

        """
        
        n_layers = min(n_layers, self.n_layers)
        rs = xs.clone()

        neglogfxs = torch.zeros(rs.shape[0])

        for i in range(n_layers):
            
            zs = self.sirts[i]._eval_rt(rs)
            neglogsirts = self.sirts[i]._eval_potential(rs, subset)

            rs = self.reference.invert_cdf(zs)
            neglogrefs = self.reference.eval_potential(rs)[0]
            neglogfxs += neglogsirts - neglogrefs

        neglogrefs = self.reference.eval_potential(rs)[0]
        neglogfxs += neglogrefs

        return rs, neglogfxs
    
    def _eval_irt_reference(
        self, 
        rs: Tensor, 
        n_layers: int = torch.inf,
        subset: str | None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the deep inverse Rosenblatt transport.

        Parameters
        ----------
        rs:
            An $n \times k$ matrix containing samples distributed 
            according to the reference density.
        n_layers: 
            The number of layers of the deep inverse Rosenblatt 
            transport to pull the samples back under. If not specified,
            the samples will be pulled back through all the layers.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        xs: 
            An $n \times k$ matrix containing the corresponding samples 
            after applying the deep inverse Rosenblatt transport.
        neglogfxs:
            An $n$-dimensional vector containing the potential function
            of the pullback of the reference density under the current 
            composition of mappings, evaluated at each sample in `xs`.

        """

        n_layers = min(n_layers, self.n_layers)
        xs = rs.clone()

        neglogfxs = self.reference.eval_potential(xs)[0]

        for i in range(n_layers-1, -1, -1):
            neglogrefs = self.reference.eval_potential(xs)[0]
            zs = self.reference.eval_cdf(xs)[0]
            xs, neglogsirts = self.sirts[i]._eval_irt(zs, subset)
            neglogfxs += neglogsirts - neglogrefs

        return xs, neglogfxs

    def eval_irt_pullback(
        self,
        potential: Callable[[Tensor], Tensor],
        rs: Tensor, 
        subset: str = "first"
    ) -> Tensor:
        r"""Evaluates the pullback, T^{\sharp}f(r), of the target 
        function under the DIRT mapping.

        This function evaluates T^{\sharp}f(r), where T denotes the 
        inverse Rosenblatt transport and f denotes a (possibly 
        unnormalised) density function.

        Parameters
        ----------
        potential:
            A function that returns the negative logarithm of the 
            function, f, to evaluate the pullback of.
        rs:
            An n * d tensor containing a set of samples from the 
            reference domain.

        Returns
        -------
        neglogTfrs
            An n-dimensional vector containing the negative logarithm 
            of the pullback function evaluated at each element in rs.
        
        """
        neglogrefs = self.reference.eval_potential(rs)[0]
        ms, neglogfms_dirt = self.eval_irt(rs, subset=subset)
        neglogfms = potential(ms)
        neglogTfrs = neglogfms + neglogrefs - neglogfms_dirt
        return neglogTfrs
    
    def eval_cirt_pullback(
        self, 
        potential: Callable[[Tensor], Tensor],
        ms: Tensor,
        rs: Tensor,
        subset: str = "first"
    ) -> Tensor:
        r"""TODO: write docstring.
        
        Potential needs to return $f(x|y)$ (adjust notation).
        
        """
        neglogrefs = self.reference.eval_potential(rs)[0]
        ms, neglogfms_cirt = self.eval_cirt(ms, rs, subset=subset)
        neglogfms = potential(ms)
        neglogTfrs = neglogfms + neglogrefs - neglogfms_cirt
        return neglogTfrs

    def eval_potential(
        self, 
        ms: Tensor,
        n_layers: Tensor = torch.inf,
        subset: str | None = None
    ) -> Tensor:
        r"""Evaluates the potential function.
        
        Returns the joint potential function, or the marginal potential 
        function for the first $k$ variables or the last $k$ variables,
        corresponding to the pullback of the reference measure under a 
        given number of layers of the DIRT.
        
        Parameters
        ----------
        ms:
            An $n \times k$ matrix containing a set of samples drawn 
            from the current DIRT approximation to the target density.
        n_layers:
            The number of layers of the current DIRT construction to
            use.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        neglogfxs:
            An $n$-dimensional vector containing the potential function
            of the target density evaluated at each element in `xs`.

        """
        n_layers = min(n_layers, self.n_layers)
        neglogfms = self.eval_rt(ms, n_layers, subset)[1]
        return neglogfms
    
    def eval_pdf(
        self, 
        ms: Tensor,
        n_layers: Tensor = torch.inf,
        subset: str | None = None
    ) -> Tensor: 
        r"""Evaluates the density function.
        
        Returns the joint density function, or the marginal density 
        function for the first $k$ variables or the last $k$ variables,
        corresponding to the pullback of the reference measure under 
        a given number of layers of the DIRT.
        
        Parameters
        ----------
        ms:
            An $n \times k$ matrix containing a set of samples drawn 
            from the DIRT approximation to the target density.
        n_layers:
            The number of layers of the current DIRT construction to 
            use.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        fms:
            An $n$-dimensional vector containing the value of the 
            approximation to the target density evaluated at each 
            element in `ms`.
        
        """
        neglogfms = self.eval_potential(ms, n_layers, subset)
        fms = torch.exp(-neglogfms)
        return fms

    def eval_rt(
        self,
        ms: Tensor,
        n_layers: Tensor = torch.inf,
        subset: str | None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the deep Rosenblatt transport.
        
        Parameters
        ----------
        ms:
            An $n \times k$ matrix of random variables drawn from the 
            density defined by the current DIRT.
        n_layers:
            The number of layers of the deep inverse Rosenblatt 
            transport to push the samples forward under. If not 
            specified, the samples will be pushed forward through all 
            the layers.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        rs:
            An $n \times k$ matrix containing the composition of 
            mappings evaluated at each value of `ms`.
        neglogfms:
            An $n$-dimensional vector containing the potential function 
            of the pullback of the reference density under the current 
            composition of mappings, evaluated at each sample in `ms`.

        """
        n_layers = min(n_layers, self.n_layers)
        neglogabsdet_ms = self.preconditioner.neglogdet_Q_inv(ms)
        xs = self.preconditioner.Q_inv(ms)
        rs, neglogfxs = self._eval_rt_reference(xs, n_layers, subset)
        neglogfms = neglogfxs + neglogabsdet_ms
        return rs, neglogfms

    def eval_irt(
        self, 
        rs: Tensor, 
        n_layers: int = torch.inf,
        subset: str | None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the deep inverse Rosenblatt transport.

        Parameters
        ----------
        rs:
            An $n \times k$ matrix containing samples distributed 
            according to the reference density.
        n_layers: 
            The number of layers of the deep inverse Rosenblatt 
            transport to pull the samples back under. If not specified,
            the samples will be pulled back through all the layers.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        ms:
            An $n \times k$ matrix containing the corresponding samples 
            from the approximation domain, after applying the deep 
            inverse Rosenblatt transport.
        neglogfms:
            An $n$-dimensional vector containing the potential function
            of the pullback of the reference density under the current 
            composition of mappings, evaluated at each sample in `xs`.

        """
        xs, neglogfxs = self._eval_irt_reference(rs, n_layers, subset)
        ms = self.preconditioner.Q(xs)
        neglogabsdet_ms = self.preconditioner.neglogdet_Q_inv(ms)
        neglogfms = neglogfxs + neglogabsdet_ms
        return ms, neglogfms
    
    def eval_cirt(
        self, 
        ms: Tensor, 
        rs: Tensor, 
        n_layers: int = torch.inf,
        subset: str | None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the conditional inverse Rosenblatt transport.

        Returns the conditional inverse Rosenblatt transport evaluated
        at a set of samples in the approximation domain. 
        
        The conditional inverse Rosenblatt transport takes the form
        
        $$
            Y|M = \mathcal{T}(\mathcal{R}_{k}(M), R),
        $$

        where $M$ is a $k$-dimensional random variable, $R$ is a 
        $n-k$-dimensional reference random variable, 
        $\mathcal{R}(\,\cdot\,)$ denotes the (full) Rosenblatt 
        transport, $\mathcal{T}(\,\cdot\,) := \mathcal{R}^{-1}(\,\cdot\,)$, 
        denotes its inverse, and $\mathcal{R}_{k}(\,\cdot\,)$ denotes 
        the Rosenblatt transport for the first (or last) $k$ variables.
        
        Parameters
        ----------
        ms:
            An $n \times k$ matrix containing samples from the 
            approximation domain.
        rs:
            An $n \times (d-k)$ matrix containing samples distributed 
            according to the reference density.
        n_layers:
            The number of layers of the deep inverse Rosenblatt 
            transport to push the samples forward under. If not 
            specified, the samples will be pushed forward through all 
            the layers.
        subset: 
            Whether `ms` corresponds to the first $k$ variables 
            (`subset='first'`) of the approximation, or the last $k$ 
            variables (`subset='last'`).
        
        Returns
        -------
        ys:
            An $n \times (d-k)$ matrix containing the realisations of 
            $Y$ corresponding to the values of `rs` after applying the 
            conditional inverse Rosenblatt transport.
        neglogfys:
            An $n$-dimensional vector containing the potential function 
            of the approximation to the conditional density of 
            $Y \textbar M$ evaluated at each sample in `rs`.
    
        """
        
        ms = torch.atleast_2d(ms)
        rs = torch.atleast_2d(rs)

        n_rs, d_rs = rs.shape
        n_ms, d_ms = ms.shape

        if d_rs == 0 or d_ms == 0:
            msg = "The dimensions of both 'ms' and 'zs' must be at least 1."
            raise Exception(msg)
        
        if d_rs + d_ms != self.dim:
            msg = ("The dimensions of X and Z must sum " 
                   + "to the dimension of the approximation.")
            raise Exception(msg)
        
        if n_rs != n_ms: 
            if n_ms != 1:
                msg = "The number of samples of X and Z must be equal."
                raise Exception(msg)
            ms = ms.repeat(n_rs, 1)
        
        direction = SIRT._get_direction(subset)
        if direction == Direction.FORWARD:
            inds_m = torch.arange(d_ms)
            inds_y = torch.arange(d_ms, self.dim)
        elif direction == Direction.BACKWARD:
            inds_m = torch.arange(d_rs, self.dim)
            inds_y = torch.arange(d_rs)
        
        # Evaluate marginal RT
        rs_m, neglogfms = self.eval_rt(ms, n_layers, subset)

        # Evaluate joint RT
        rs_my = torch.empty((n_rs, self.dim))
        rs_my[:, inds_m] = rs_m 
        rs_my[:, inds_y] = rs
        mys, neglogfmys = self.eval_irt(rs_my, n_layers, subset)
        
        ys = mys[:, inds_y]
        neglogfys = neglogfmys - neglogfms

        return ys, neglogfys
    
    def eval_rt_jac(self, ms: Tensor, subset: str | None = None) -> Tensor:
        r"""Evaluates the Jacobian of the deep Rosenblatt transport.

        Evaluates the Jacobian of the mapping $R = \mathcal{R}(X)$, 
        where $R$ denotes the reference random variable and $X$ denotes 
        the approximation to the target random variable. 

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

        Returns
        -------
        Js:
            A $k \times n \times k$ tensor, where element $ijk$ 
            contains element $ik$ of the Jacobian for the $j$th sample 
            in `xs`.

        """

        n_ms, d_ms = ms.shape

        def _eval_rt(ms: Tensor) -> Tensor:
            ms = ms.reshape(n_ms, d_ms)
            return self.eval_rt(ms, subset=subset)[0].sum(dim=0)
        
        Js = jacobian(_eval_rt, ms.flatten(), vectorize=True)
        return Js.reshape(d_ms, n_ms, d_ms)
    
    def eval_irt_jac(self, rs: Tensor, subset: str | None = None) -> Tensor:
        r"""Evaluates the Jacobian of the deep inverse Rosenblatt transport.

        Evaluates the Jacobian of the mapping $X = \mathcal{R}^{-1}(R)$, 
        where $R$ denotes the reference random variable and $X$ denotes 
        the approximation to the target random variable. 

        Note that element $J_{ij}$ of the Jacobian is given by
        $$J_{ij} = \frac{\partial x_{i}}{\partial r_{j}}.$$

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

        Returns
        -------
        Js:
            A $k \times n \times k$ tensor, where element $ijk$ 
            contains element $ik$ of the Jacobian for the $j$th sample 
            in `xs`.

        """

        n_rs, d_rs = rs.shape

        def _eval_irt(rs: Tensor) -> Tensor:
            rs = rs.reshape(n_rs, d_rs)
            return self.eval_irt(rs, subset=subset)[0].sum(dim=0)
        
        Js = jacobian(_eval_irt, rs.flatten(), vectorize=True)
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
        ms:
            An $n \times d$ matrix containing the generated samples.
        
        """
        rs = self.reference.random(self.dim, n)
        ms = self.eval_irt(rs)[0]
        return ms
    
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
        ms:
            An $n \times d$ matrix containing the generated samples.

        """
        rs = self.reference.sobol(self.dim, n)
        ms = self.eval_irt(rs)[0]
        return ms
    
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
    prior:
        An object which provides a coupling between the prior and a 
        product-form reference density.
    bases:
        A list of polynomial bases (one for each dimension), or a 
        single polynomial basis (to be used in all dimensions), used to 
        construct the functional tensor trains at each iteration.
    bridge: 
        An object used to generate the intermediate densities to 
        approximate at each stage of the DIRT construction.
    tt_options:
        Options for constructing the SIRT approximation to the 
        ratio function (*i.e.*, the pullback of the current bridging 
        density under the existing composition of mappings) at each 
        iteration.
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
        
        def neglogfx_Q(xs: Tensor) -> Tensor:
            ms = self.preconditioner.Q(xs)
            neglogdets = self.preconditioner.neglogdet_Q(xs)
            neglogliks = negloglik(ms)
            neglogpris = neglogpri(ms)
            self.num_eval += xs.shape[0]
            return neglogpris + neglogliks + neglogdets

        if bridge is None:
            bridge = Tempering(min_beta=1e-3, ess_tol=0.4)
        if tt_options is None:
            tt_options = TTOptions(max_cross=1, tt_method="amen")
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

        # rs, _ = self.bases.sample_measure(self.pre_sample_size)
        # neglogrefs_rs = self.reference.eval_potential(rs)[0]
        
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
                if self.dirt_options.verbose:
                    t1 = time.time()
                    dirt_info("DIRT construction complete.")
                    dirt_info(f" • Layers: {self.n_layers}.")
                    dirt_info(f" • Total function evaluations: {self.num_eval}.")
                    dirt_info(f" • Total time: {t1-t0:.2f} s.")
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