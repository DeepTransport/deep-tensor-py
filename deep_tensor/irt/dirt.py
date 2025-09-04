import math
import time
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
from torch.autograd.functional import jacobian

from .dirt_options import DIRTOptions
from .sirt import SIRT, SUBSET2DIRECTION
from ..bridging_densities import Bridge, Tempering
from ..ftt import Direction, FTT
from ..preconditioners import Preconditioner
from ..target_functions import TargetFunc
from ..tools.printing import dirt_info, format_time
from ..tools import compute_f_divergence


from torch import Tensor 


class DIRT():
    r"""Deep (squared) inverse Rosenblatt transport.

    Parameters
    ----------
    target_func:
        The density function to be approximated.
    preconditioner:
        An initial guess as to the mappings between the reference 
        random variable and the target random variable.
    ftt:
        A functional tensor train object (used to construct each layer 
        of the DIRT).
    bridge: 
        An object used to generate the ratio functions to approximate 
        at each layer of the DIRT construction.
    options: 
        Options which control the DIRT construction process.
    
    """

    def __init__(
        self, 
        target_func: TargetFunc,
        preconditioner: Preconditioner,
        ftt: FTT, 
        bridge: Bridge | None = None,
        options: DIRTOptions | None = None
    ):
        # TODO: need to reset the bridge prior to starting. Ideally we 
        # should be able to use the same bridge object to build 
        # multiple DIRT objects.
        # TODO: need to make sure samples are available if the adaptive 
        # bridges are going to be used.

        if not isinstance(target_func, TargetFunc):
            target_func = TargetFunc(target_func)
        if bridge is None:
            bridge = Tempering()
        if options is None:
            options = DIRTOptions()
        
        self.target_func = target_func
        self.preconditioner = preconditioner
        self.dim = preconditioner.dim
        self.reference = preconditioner.reference
        self.domain = self.reference.domain
        self.ftt = ftt
        self.bridge = bridge
        self.bridge.initialise(preconditioner, target_func)

        self.ratio_type = options.ratio_type 
        self.num_error_samples = options.num_error_samples
        self.defensive = options.defensive
        self.cdf_tol = options.cdf_tol
        self.verbose = options.verbose

        # self.num_eval_diagnostics = 0
        
        self.sirts: Dict[int, SIRT] = {}

        self._build()
        return
    
    @property 
    def num_layers(self) -> int:
        return self.bridge.n_layers
    
    @num_layers.setter
    def num_layers(self, value: int) -> None:
        self.bridge.n_layers = value 
        return

    @property
    def num_eval(self) -> int:
        n = sum([sirt.num_eval for sirt in self.sirts.values()])
        # TODO: need to add on the debugging samples to this.
        return n
    
    @property
    def log_z(self) -> float:
        if not self.sirts.keys():
            return 0.0
        return sum([math.log(self.sirts[k].z) for k in self.sirts])
    
    def eval_target_pullback(self, us: Tensor) -> Tensor:
        """Evaluates the pullback of the target density function under 
        the preconditioning mapping, at a set of samples in the 
        reference domain.
        """
        xs = self.preconditioner.Q(us)
        neglogdets = self.preconditioner.neglogdet_Q(us)
        neglogfxs = self.target_func(xs)
        # self.num_eval += us.shape[0]
        neglogfus = neglogfxs + neglogdets
        return neglogfus
  
    def eval_ratio_func(self, rs: Tensor) -> Tensor:
        """Evaluates the current ratio function at each element in rs, 
        where rs is a set of samples from the reference domain.
        """
        us, neglogfus_dirt = self._eval_irt_reference(rs)
        neglogratios = self.bridge.ratio_func(self.ratio_type, rs, us, neglogfus_dirt)
        return neglogratios

    def _get_new_layer(self) -> SIRT:
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
        if self.num_layers == 0:
            ftt = self.ftt.clone()
        else:
            ftt = self.sirts[self.num_layers-1].ftt.clone()
        sirt = SIRT(
            self.eval_ratio_func, 
            ftt, 
            self.reference, 
            self.domain, 
            self.defensive, 
            self.cdf_tol
        )
        return sirt

    def _print_progress(
        self,
        log_weights: Tensor, 
        neglogfus: Tensor, 
        neglogfus_dirt: Tensor,
        cum_time: float
    ) -> None:

        msg = [
            f"Iter: {self.num_layers+1:=2}",
            f"Cum. Fevals: {self.num_eval:=.2e}",
            f"Cum. Time: {cum_time:=.2e} s"
        ]
        msg += self.bridge._get_diagnostics(log_weights, neglogfus, neglogfus_dirt)
        dirt_info(" | ".join(msg))
        return
    
    def _build(self) -> None:
        """Constructs the DIRT object to approximate the target function."""

        t0 = time.time()
        
        while True:

            rs = self.reference.random(self.dim, self.num_error_samples)
            us, neglogfus_dirt = self._eval_irt_reference(rs)

            log_weights, neglogbridges = self.bridge.update(us, neglogfus_dirt)

            if self.verbose > 0:
                cum_time = time.time() - t0
                self._print_progress(log_weights, neglogbridges, 
                                     neglogfus_dirt, cum_time)

            self.sirts[self.num_layers] = self._get_new_layer()
            self.num_layers += 1

            if self.bridge.is_last:
                break

        if self.verbose:

            # Note: the Hellinger divergence is invariant to bijective 
            # transformations
            rs = self.reference.random(self.dim, self.num_error_samples)
            us, neglogfus_dirt = self._eval_irt_reference(rs)
            neglogfus = self.eval_target_pullback(us) # TODO: the bridge should probably handle this...
            dhell2 = compute_f_divergence(-neglogfus_dirt, -neglogfus)

            t1 = time.time()
            
            dirt_info("DIRT construction complete.")
            dirt_info(f" • Layers: {self.num_layers}.")
            dirt_info(f" • Total function evaluations: {self.num_eval:,}.")
            dirt_info(f" • Total time: {format_time(t1-t0)}.")
            dirt_info(f" • DHell: {dhell2.sqrt():.4f}.")
        
        return
    
    def _eval_rt_reference(
        self,
        us: Tensor,
        subset: str,
        num_layers: int
    ) -> Tuple[Tensor, Tensor]:
        """Evaluates the deep Rosenblatt transport for the pullback of 
        the target density under the preconditioning map.
        """
        
        rs = us.clone()
        neglogfus = torch.zeros(rs.shape[0])

        for i in range(num_layers):
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
        num_layers: int | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Evaluates the deep inverse Rosenblatt transport for the 
        pullback of the target density under the preconditioning map.
        """

        if num_layers is None:
            num_layers = self.num_layers
        
        us = rs.clone()
        neglogfus = self.reference.eval_potential(us)[0]

        for i in range(num_layers-1, -1, -1):
            neglogrefs = self.reference.eval_potential(us)[0]
            zs = self.reference.eval_cdf(us)[0]
            us, neglogsirts = self.sirts[i]._eval_irt(zs, subset)
            neglogfus += neglogsirts - neglogrefs

        return us, neglogfus

    def _parse_subset(self, subset: str | None) -> str:
        
        if subset is None:
            subset = "first"
        subset = subset.lower()

        if subset == "last" and self.num_layers > 1:
            msg = (
                "When using a DIRT object with more than one layer, it "
                "is not possible to sample from the marginal densities "
                "in the final k variables (where k < d) or the density "
                "of the first (d-k) variables conditioned on the final "
                "k variables. Please reverse the variable ordering, or "
                "construct a DIRT object with a single layer."
            )
            raise Exception(msg)
        if subset not in ("first", "last"):
            msg = (
                f"Invalid subset parameter encountered (subset='{subset}'). "
                "Valid choices are 'first', 'last'."
            )
            raise ValueError(msg)
        
        return subset

    def eval_rt(
        self,
        xs: Tensor,
        subset: str | None = None,
        num_layers: int | None = None
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
        num_layers:
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
        if num_layers is None:
            num_layers = self.num_layers
        subset = self._parse_subset(subset)
        neglogdet_xs = self.preconditioner.neglogdet_Q_inv(xs, subset)
        us = self.preconditioner.Q_inv(xs, subset)
        rs, neglogfus = self._eval_rt_reference(us, subset, num_layers)
        neglogfxs = neglogfus + neglogdet_xs
        return rs, neglogfxs

    def eval_irt(
        self, 
        rs: Tensor, 
        subset: str | None = None,
        num_layers: int | None = None
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
        num_layers: 
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
        if num_layers is None:
            num_layers = self.num_layers
        subset = self._parse_subset(subset)
        us, neglogfus = self._eval_irt_reference(rs, subset, num_layers)
        xs = self.preconditioner.Q(us, subset)
        neglogdet_xs = self.preconditioner.neglogdet_Q_inv(xs, subset)
        neglogfxs = neglogfus + neglogdet_xs
        return xs, neglogfxs
    
    def eval_cirt(
        self, 
        ys: Tensor, 
        rs: Tensor, 
        subset: str = "first",
        num_layers: int | None = None
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
        num_layers:
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
        else:
            inds_y = torch.arange(d_rs, self.dim)
            inds_x = torch.arange(d_rs)
        
        # Evaluate marginal RT
        rs_y, neglogfys = self.eval_rt(ys, subset, num_layers)

        # Evaluate joint RT
        rs_yx = torch.empty((n_rs, self.dim))
        rs_yx[:, inds_y] = rs_y 
        rs_yx[:, inds_x] = rs
        yxs, neglogfyxs = self.eval_irt(rs_yx, subset, num_layers)
        
        xs = yxs[:, inds_x]
        neglogfxs = neglogfyxs - neglogfys
        return xs, neglogfxs

    def eval_irt_pullback(
        self,
        potential: Callable[[Tensor], Tensor],
        rs: Tensor, 
        subset: str | None = None,
        num_layers: int | None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the pullback of a density function.

        This function evaluates $\mathcal{T}^{\sharp}f(r)$, where 
        $\mathcal{T}(\cdot)$ denotes the inverse Rosenblatt transport 
        and $f(\cdot)$ denotes an arbitrary density function.

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
        num_layers: 
            The number of layers of the deep inverse Rosenblatt 
            transport to pull the samples back under. If not specified,
            the samples will be pulled back through all the layers.

        Returns
        -------
        neglogTfrs:
            An $n$-dimensional vector containing the potential of the 
            pullback function evaluated at each element in `rs`; that 
            is, $-\log(\mathcal{T}^{\sharp}f(r))$.
        neglogfxs:
            An $n$-dimensional vector containing the potential of the 
            target function evaluated at each element in `rs`, pushed 
            forward under the IRT; that is, $-\log(f(\mathcal{T}(r)))$.
        
        """
        neglogrefs = self.reference.eval_potential(rs)[0]
        xs, neglogfxs_irt = self.eval_irt(rs, subset, num_layers)
        neglogfxs = potential(xs)
        neglogTfrs = neglogfxs + neglogrefs - neglogfxs_irt
        return neglogTfrs, neglogfxs
    
    def eval_cirt_pullback(
        self, 
        potential: Callable[[Tensor], Tensor],
        ys: Tensor,
        rs: Tensor,
        subset: str = "first",
        num_layers: int | None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the pullback of a conditional density function.

        This function evaluates $\mathcal{T}^{\sharp}f(r\|y)$, where 
        $\mathcal{T}(\cdot)$ denotes the inverse Rosenblatt transport 
        and $f(\cdot\|y)$ denotes an arbitrary conditional density 
        function.

        Parameters
        ----------
        potential:
            A function that takes an $n \times (d-k)$ matrix of samples 
            from the approximation domain, and returns an 
            $n$-dimensional vector containing the potential function 
            associated with $f(\cdot\|y)$ evaluated at each sample.
        ys:
            A matrix containing samples from the approximation domain.
            The matrix should have dimensions $1 \times k$ (if the same 
            realisation of $Y$ is to be used for all samples in `rs`) 
            or $n \times k$ (if a different realisation of $Y$ is to be 
            used for each samples in `rs`).
        rs:
            An $n \times (d-k)$ matrix containing a set of samples from 
            the reference domain.
        subset: 
            Whether `ys` corresponds to the first $k$ variables 
            (`subset='first'`) of the approximation, or the last $k$ 
            variables (`subset='last'`).
        num_layers: 
            The number of layers of the deep inverse Rosenblatt 
            transport to pull the samples back under. If not specified,
            the samples will be pulled back through all the layers.

        Returns
        -------
        neglogTfrs:
            An $n$-dimensional vector containing the potential of the 
            pullback function evaluated at each element in `rs`; that 
            is, $-\log(\mathcal{T}^{\sharp}f(r\|y))$.
        neglogfxs:
            An $n$-dimensional vector containing the potential of the 
            target function evaluated at each element in `rs`, pushed 
            forward under the IRT; that is, $-\log(f(\mathcal{T}(r)\|y))$.
        
        """
        neglogrefs = self.reference.eval_potential(rs)[0]
        xs, neglogfxs_cirt = self.eval_cirt(ys, rs, subset, num_layers)
        neglogfxs = potential(xs)
        neglogTfrs = neglogfxs + neglogrefs - neglogfxs_cirt
        return neglogTfrs, neglogfxs

    def eval_potential(
        self, 
        xs: Tensor,
        subset: str | None = None,
        num_layers: int | None = None
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
        num_layers:
            The number of layers of the current DIRT construction to
            use when computing the potential. If not specified, all 
            layers will be used when computing the potential.

        Returns
        -------
        neglogfxs:
            An $n$-dimensional vector containing the potential function
            of the target density evaluated at each element in `xs`.

        """
        neglogfxs = self.eval_rt(xs, subset, num_layers)[1]
        return neglogfxs
    
    def eval_pdf(
        self, 
        xs: Tensor,
        subset: str | None = None,
        num_layers: int | None = None
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
        num_layers:
            The number of layers of the current DIRT construction to 
            use. If not specified, all 

        Returns
        -------
        fxs:
            An $n$-dimensional vector containing the value of the 
            approximation to the target density evaluated at each 
            element in `xs`.
        
        """
        neglogfxs = self.eval_potential(xs, subset, num_layers)
        fxs = torch.exp(-neglogfxs)
        return fxs

    def eval_potential_cond(
        self, 
        ys: Tensor, 
        xs: Tensor, 
        subset: str = "first",
        num_layers: int | None = None
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
        num_layers:
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
        else:
            yxs = torch.hstack((xs, ys))
        
        # Evaluate marginal RT
        neglogfys = self.eval_potential(ys, subset, num_layers)
        neglogfyxs = self.eval_potential(yxs, subset, num_layers)

        neglogfxs = neglogfyxs - neglogfys
        return neglogfxs

    def eval_rt_jac(
        self, 
        xs: Tensor, 
        subset: str | None = None,
        num_layers: int | None = None 
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
        num_layers: 
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
            return self.eval_rt(xs, subset, num_layers)[0].sum(dim=0)
        
        Js: Tensor = jacobian(_eval_rt, xs.flatten(), vectorize=True)
        return Js.reshape(d_xs, n_xs, d_xs)
    
    def eval_irt_jac(
        self, 
        rs: Tensor, 
        subset: str | None = None,
        num_layers: int | None = None 
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
        num_layers: 
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
            return self.eval_irt(rs, subset, num_layers)[0].sum(dim=0)
        
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
    

class DIRTMapping(Preconditioner):
    r"""A preconditioning mapping constructed using a previously 
    constructed DIRT.

    Parameters
    ----------
    dirt: 
        A previously constructed DIRT object.
    
    TODO: it could make sense to have a function which returns Q and 
    neglogdet_Q together, etc. Otherwise the RT/IRT functions will be 
    called 2x more than necessary.
    """

    def __init__(self, dirt: DIRT):
        self.dirt = dirt
        self.reference = dirt.reference
        self.dim = dirt.dim
        return

    def Q(self, us: Tensor, subset: str = "first") -> Tensor:
        return self.dirt.eval_irt(us, subset)[0]
    
    def Q_inv(self, xs: Tensor, subset: str = "first") -> Tensor:
        return self.dirt.eval_rt(xs, subset)[0]
    
    def neglogdet_Q(self, us: Tensor, subset: str = "first") -> Tensor:
        neglogrefs = self.reference.eval_potential(us)[0]
        neglogfxs = self.dirt.eval_irt(us, subset)[1]
        return neglogrefs - neglogfxs
    
    def neglogdet_Q_inv(self, xs: Tensor, subset: str = "first") -> Tensor: 
        us, neglogfus = self.dirt.eval_rt(xs, subset)
        neglogrefs = self.reference.eval_potential(us)[0]
        return neglogfus - neglogrefs