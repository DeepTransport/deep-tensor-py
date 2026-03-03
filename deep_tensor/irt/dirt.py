import math
import time
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor
from torch.autograd.functional import jacobian

from .dirt_options import DIRTOptions
from .sirt import SIRT, SUBSET2DIRECTION
from ..bridging_densities import Bridge, Tempering
from ..ftt import Direction, FTT
from ..preconditioners import Preconditioner
from ..subspaces import Subspace, IdentitySubspace
from ..target_functions import TargetFunc
from ..tools.printing import dirt_info, format_time
from ..tools import compute_f_divergence

from torch import Tensor 


def eval_normal_potential(xs: Tensor) -> Tensor:
    """Evalutes the potential function of the (normalised) unit normal 
    density at a set of values.
    """
    xs = torch.atleast_2d(xs)
    dim = xs.shape[1]
    neglogfxs = (
        0.5 * xs.square().sum(dim=1)
        + 0.5 * dim * math.log(2.0*torch.pi)
    )
    return neglogfxs


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
        subspace: Subspace | None = None,
        options: DIRTOptions | None = None,
        device: torch.device = torch.get_default_device()
    ):

        if not isinstance(target_func, TargetFunc):
            target_func = TargetFunc(target_func)
        if subspace is None:
            subspace = IdentitySubspace(preconditioner.dim)
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
        self.subspace = subspace
        self.subspaces = {-1: subspace}
        self.ratio_type = options.ratio_type 
        self.num_error_samples = options.num_error_samples
        self.defensive = options.defensive
        self.cdf_tol = options.cdf_tol
        self.verbose = options.verbose
        self.sirts: Dict[int, SIRT] = {}
        self.device = device

        if self.bridge.is_adaptive and self.num_error_samples == 0:
            msg = (
                "The bridging densities are being chosen adaptively, "
                "which requires a non-zero number of error samples. "
                "Either pre-specify the bridging densities or set "
                "num_error_samples to a positive number (ideally at "
                "least 100)."
            )
            raise Exception(msg)

        self._build()
        return
    
    @property 
    def num_layers(self) -> int:
        return self.bridge.num_layers
    
    @num_layers.setter
    def num_layers(self, value: int) -> None:
        self.bridge.num_layers = value 
        return

    @property 
    def num_eval_sirt(self) -> int:
        return sum([self.sirts[k].num_eval for k in self.sirts])
    
    @property 
    def num_eval_diagnostic(self) -> int:
        return self.num_error_samples * (self.bridge.num_layers + 1)

    @property
    def num_eval(self) -> int:
        return self.num_eval_sirt + self.num_eval_diagnostic
    
    @property 
    def num_eval_construction(self) -> int:
        return sum([self.sirts[k].num_eval_construction for k in self.sirts])
    
    @property
    def log_z(self) -> float:
        if not self.sirts.keys():
            return 0.0
        return sum([math.log(self.sirts[k].z) for k in self.sirts])
    
    @property 
    def subspace_dims(self) -> List:
        return [self.subspaces[k].dim_red for k in range(self.num_layers)]
    
    @property
    def dhell_ratios_red(self) -> List:
        """The Hellinger divergences between the reduced ratio functions 
        and their DIRT approximations.
        """
        return [self.sirts[k].dhell_ratio for k in range(self.num_layers)]
    
    @property 
    def error_accs(self) -> List:
        return [self.subspaces[k].error_acc for k in range(self.num_layers)]  # type: ignore
    
    @property 
    def error_news(self) -> List:
        return [self.subspaces[k].error_new for k in range(self.num_layers)]  # type: ignore
  
    def eval_ratio_func(self, rs: Tensor) -> Tensor:
        """Evaluates the current ratio function at each element in rs, 
        where rs is a set of samples from the reference domain.
        """
        us, neglogfus_dirt = self._eval_irt_reference(rs)
        neglogratios = self.bridge.ratio_func(self.ratio_type, 
                                              rs, us, neglogfus_dirt)
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
        
        if isinstance(self.subspace, IdentitySubspace) and self.num_layers > 0:
            ftt = self.sirts[self.num_layers-1].ftt.clone()
        else:
            # Cannot use the previous FTT due to the possible change in 
            # dimension
            ftt = self.ftt.clone()
    
        self.subspaces[self.num_layers] = self.subspaces[self.num_layers-1].clone()

        # TODO: compute the diagnostics at the same time as updating 
        # the subspace..?

        self.subspaces[self.num_layers].update(
            self.bridge.eval_gradneglog, # type: ignore
            self._eval_irt_reference
        )

        def target_func(xs):
            # TODO: maybe increment the number of function evaluations in here.
            return self.subspaces[self.num_layers].eval_neglogprofile(self.eval_ratio_func, xs)

        sirt = SIRT(
            target_func, 
            ftt, 
            self.subspaces[self.num_layers].dim_red,
            self.reference, 
            self.defensive, 
            self.cdf_tol,
            device=self.device
        )

        self.sirts[self.num_layers] = sirt

        rs = self.reference.random(n=1000, d=self.dim)
        us, neglogratios_dirt = self._eval_irt_reference_k(rs, self.num_layers, subset="first")
        neglogratios_exact = self.eval_ratio_func(us)
        dhell_ratio = compute_f_divergence(-neglogratios_dirt, -neglogratios_exact).sqrt()
        self.dhell_ratios.append(dhell_ratio)
        return sirt

    def _print_progress(
        self,
        log_weights: Tensor | None, 
        neglogfus: Tensor | None, 
        neglogfus_dirt: Tensor | None,
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

        self.dhell_bridges = []
        self.dhell_targets = []
        self.dhell_ratios = []
        
        while True:
            
            if self.num_error_samples > 0:
                rs = self.reference.random(self.num_error_samples, self.dim, device=self.device)
                us, neglogfus_dirt = self._eval_irt_reference(rs)
                log_weights, neglogbridges = self.bridge.update(us, neglogfus_dirt)

                neglogfus_target = self.bridge._eval_pullback(us)
                dhell_bridge = compute_f_divergence(-neglogfus_dirt, -neglogbridges).sqrt()
                dhell_target = compute_f_divergence(-neglogfus_dirt, -neglogfus_target).sqrt()

            else:
                log_weights, neglogbridges, neglogfus_dirt = None, None, None
                dhell_bridge, dhell_target = None, None

            if self.bridge.num_layers > 0:
                self.dhell_bridges.append(dhell_bridge)
                self.dhell_targets.append(dhell_target)

            if self.verbose > 0:
                cum_time = time.time() - t0
                self._print_progress(log_weights, neglogbridges, neglogfus_dirt, cum_time)

            self._get_new_layer()
            self.num_layers += 1
            if self.bridge.is_last:
                break

        if self.verbose:

            info_msgs = [
                "DIRT construction complete.",
                f"Layers: {self.num_layers}.",
                f"Total function evaluations: {self.num_eval:,}."
            ]

            if self.num_error_samples > 0:
                # Note: the Hellinger divergence is invariant to bijective 
                # transformations.
                rs = self.reference.random(self.num_error_samples, self.dim, device=self.device)
                us, neglogfus_dirt = self._eval_irt_reference(rs)
                neglogfus = self.bridge._eval_pullback(us)
                dhell = compute_f_divergence(-neglogfus_dirt, -neglogfus).sqrt() 
                info_msgs += [f"DHell: {dhell:.4f}."]
                self.dhell_bridges.append(dhell)  # TODO: fix this. it should be the smoothed function.
                self.dhell_targets.append(dhell)

            t1 = time.time()
            info_msgs += [f"Total time: {format_time(t1-t0)}."]
            
            for msg in info_msgs:
                dirt_info(f" • {msg}")
        
        return

    def _eval_rt_reference_k(
        self,
        us: Tensor,
        i: int,
        subset: str
    ) -> Tuple[Tensor, Tensor]:
        """TODO: write this.."""

        us_red = self.subspaces[i].eval_red2coef(us)
        us_comp = self.subspaces[i].eval_comp2coef(us)

        zs_red = self.sirts[i]._eval_rt(us_red, subset)
        neglogfrs_red = self.sirts[i]._eval_potential(us_red, subset)
        rs_red = self.reference.invert_cdf(zs_red)
        rs_red = self.subspaces[i].eval_coef2red(rs_red)

        rs_comp = self.subspaces[i].eval_coef2comp(us_comp)
        neglogfrs_comp = eval_normal_potential(us_comp)

        rs = rs_red + rs_comp 
        neglogfrs = neglogfrs_red + neglogfrs_comp

        return rs, neglogfrs
    
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
        neglogfus = torch.zeros(rs.shape[0], device=self.device)

        for i in range(num_layers):
            rs, neglogsirts = self._eval_rt_reference_k(rs, i, subset)
            neglogrefs = self.reference.eval_potential(rs)[0]
            neglogfus += neglogsirts - neglogrefs

        neglogrefs = self.reference.eval_potential(rs)[0]
        neglogfus += neglogrefs

        return rs, neglogfus
    
    def _eval_irt_reference_k(
        self, 
        rs: Tensor, 
        i: int, 
        subset: str
    ) -> Tuple[Tensor, Tensor]:
        """TODO: write docstring.
        
        this function evaluates the reduced SIRT..
        """
        
        rs_red = self.subspaces[i].eval_red2coef(rs)
        rs_comp = self.subspaces[i].eval_comp2coef(rs)

        zs_red = self.reference.eval_cdf(rs_red)[0]
        ws_red, neglogfus_red = self.sirts[i]._eval_irt(zs_red, subset)
        us_red = self.subspaces[i].eval_coef2red(ws_red)
        
        us_comp = self.subspaces[i].eval_coef2comp(rs_comp)
        neglogfus_comp = eval_normal_potential(rs_comp)

        us = us_red + us_comp
        neglogfus = neglogfus_red + neglogfus_comp

        return us, neglogfus

    def _eval_irt_reference(
        self, 
        rs: Tensor, 
        subset: str = "first",
        num_layers: int | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Evaluates the deep inverse Rosenblatt transport for the 
        pullback of the target density under the preconditioning map.

        TODO: if a non-identity subspace is used, we lose the ability 
        to evaluate marginals etc. need to make this clear in the code.
        """

        if num_layers is None:
            num_layers = self.num_layers
        
        us = rs.clone()
        neglogfus = self.reference.eval_potential(us)[0]

        for i in range(num_layers-1, -1, -1):
            neglogrefs = self.reference.eval_potential(us)[0]
            us, neglogsirts = self._eval_irt_reference_k(us, i, subset)

            # xxs = torch.vstack([self._eval_rt_reference_k(us[k, :], i, subset)[0] for k in range(us.shape[0])])
            # yys = torch.vstack([self._eval_irt_reference_k(xxs[k, :], i, subset)[0] for k in range(us.shape[0])])

            # TODO: could try adding eval_rt_ref_k in here and see what happens..

            # zs = self.reference.eval_cdf(us)[0]
            # us, neglogsirts = self.sirts[i]._eval_irt(zs, subset)
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
        
        xs = xs.to(self.device)
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
        
        rs = rs.to(self.device)
        rs = self.reference._project_to_domain(rs)

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

        ys = ys.to(self.device)
        rs = rs.to(self.device)
        
        ys = torch.atleast_2d(ys)
        rs = torch.atleast_2d(rs)
        n_rs, d_rs = rs.shape
        n_ys, d_ys = ys.shape
        rs = self.reference._project_to_domain(rs)

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
            inds_y = torch.arange(d_ys, device=self.device)
            inds_x = torch.arange(d_ys, self.dim, device=self.device)
        else:
            inds_y = torch.arange(d_rs, self.dim, device=self.device)
            inds_x = torch.arange(d_rs, device=self.device)
        
        # Evaluate marginal RT
        rs_y, neglogfys = self.eval_rt(ys, subset, num_layers)

        # Evaluate joint RT
        rs_yx = torch.empty((n_rs, self.dim), device=self.device)
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
        xs:
            An $n \times d$ matrix containing the inverse Rosenblatt 
            transport evaluated at each element in `rs`.
        neglogTfrs:
            An $n$-dimensional vector containing the potential of the 
            pullback function evaluated at each element in `rs`; that 
            is, $-\log(\mathcal{T}^{\sharp}f(r))$.
        neglogfxs:
            An $n$-dimensional vector containing the potential of the 
            target function evaluated at each element in `rs`, pushed 
            forward under the IRT; that is, $-\log(f(\mathcal{T}(r)))$.
        
        """
        rs = rs.to(self.device)
        rs = self.reference._project_to_domain(rs)
        neglogrefs = self.reference.eval_potential(rs)[0]
        xs, neglogfxs_irt = self.eval_irt(rs, subset, num_layers)
        neglogfxs = potential(xs)
        neglogTfrs = neglogfxs + neglogrefs - neglogfxs_irt
        return xs, neglogTfrs, neglogfxs
    
    def eval_cirt_pullback(
        self, 
        potential: Callable[[Tensor], Tensor],
        ys: Tensor,
        rs: Tensor,
        subset: str = "first",
        num_layers: int | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
        xs:
            An $n \times d$ matrix containing the inverse Rosenblatt 
            transport evaluated at each element in `rs`.
        neglogTfrs:
            An $n$-dimensional vector containing the potential of the 
            pullback function evaluated at each element in `rs`; that 
            is, $-\log(\mathcal{T}^{\sharp}f(r\|y))$.
        neglogfxs:
            An $n$-dimensional vector containing the potential of the 
            target function evaluated at each element in `rs`, pushed 
            forward under the IRT; that is, $-\log(f(\mathcal{T}(r)\|y))$.
        
        """
        ys = ys.to(self.device)
        rs = rs.to(self.device)
        rs = self.reference._project_to_domain(rs)
        neglogrefs = self.reference.eval_potential(rs)[0]
        xs, neglogfxs_cirt = self.eval_cirt(ys, rs, subset, num_layers)
        neglogfxs = potential(xs)
        neglogTfrs = neglogfxs + neglogrefs - neglogfxs_cirt
        return xs, neglogTfrs, neglogfxs

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
            use. If not specified, all layers will be used when 
            evaluating the density function.

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

        ys = ys.to(self.device)
        xs = xs.to(self.device)
        
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

        if SUBSET2DIRECTION[subset] == Direction.FORWARD:
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

        xs = xs.to(self.device)
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

        rs = rs.to(self.device)
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
        rs = self.reference.random(n, self.dim, device=self.device)
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
        rs = self.reference.sobol(n, self.dim, device=self.device)
        xs = self.eval_irt(rs)[0]
        return xs
    

class DIRTMapping(Preconditioner):
    r"""A preconditioning mapping constructed using a previously 
    constructed DIRT.

    Parameters
    ----------
    dirt: 
        A previously constructed DIRT object.
    
    """

    # TODO: it could make sense to have a function which returns Q and 
    # neglogdet_Q together, etc. Otherwise the RT/IRT functions will be 
    # called 2x more than necessary.

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