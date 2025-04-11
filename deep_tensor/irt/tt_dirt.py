from copy import deepcopy
from typing import Callable, Dict, List, Tuple
import warnings

import torch
from torch import Tensor

from .tt_sirt import TTSIRT
from ..bridging_densities import Bridge, Tempering1
from ..ftt import ApproxBases, InputData
from ..options import DIRTOptions, TTOptions
from ..polynomials import Basis1D
from ..prior_transformation import PriorTransformation
from ..tools import compute_f_divergence
from ..tools.printing import dirt_info


class TTDIRT():
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
    sirt_options:
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
    Foundations of Computational Mathematics, **22**, 1863--1922.
    
    """

    def __init__(self, 
        negloglik: Callable[[Tensor], Tensor],
        prior: PriorTransformation,
        bases: Basis1D | List[Basis1D], 
        bridge: Bridge|None = None,
        sirt_options: TTOptions|None = None,
        dirt_options: DIRTOptions|None = None,
        prev_approx: Dict[int, TTSIRT]|None = None
    ):
        
        def negloglik_Q(rs: Tensor) -> Tensor:
            """Computes the negative log-likelihood of a set of samples 
            from the reference domain, by first transforming them using 
            the mapping Q.
            """
            xs = self.prior.Q(rs)
            return negloglik(xs)

        if bridge is None:
            bridge = Tempering1(min_beta=1e-3, ess_tol=0.4)
        if sirt_options is None:
            sirt_options = TTOptions(max_cross=1, tt_method="amen")
        if dirt_options is None:
            dirt_options = DIRTOptions()

        self.negloglik = negloglik_Q
        self.prior = prior
        self.reference = self.prior.reference
        self.domain = self.prior.reference.domain
        self.dim = prior.dim
        self.bases = ApproxBases(bases, self.domain, self.dim)
        self.bridge = bridge
        self.sirt_options = sirt_options
        self.dirt_options = dirt_options
        self.prev_approx = prev_approx
        self.pre_sample_size = (self.dirt_options.num_samples 
                                + self.dirt_options.num_debugs)
        self.irts: dict[int, TTSIRT] = {}
        self.num_eval = 0
        self.log_z = 0.0

        self._build()
        return

    @property 
    def n_layers(self) -> int:
        return self.bridge.n_layers
    
    @n_layers.setter
    def n_layers(self, value):
        self.bridge.n_layers = value 
        return

    def _get_potential_to_density(
        self, 
        bases: ApproxBases, 
        neglogratios: Tensor, 
        xs: Tensor
    ) -> Tensor:
        """Returns the function we aim to approximate (i.e., the 
        square-root of the ratio function divided by the weighting 
        function associated with the reference measure).

        Parameters
        ----------
        bases: 
            The bases for the reference measure.
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
        neglogwrs = bases.eval_measure_potential(xs)[0]
        log_ys = -0.5 * (neglogratios - neglogwrs)
        return torch.exp(log_ys)

    def _get_inputdata(
        self, 
        bases: ApproxBases, 
        xs: Tensor, 
        neglogratios: Tensor 
    ) -> InputData:
        """Generates a set of input data and debugging samples used to 
        initialise DIRT.
        
        Parameters
        ----------
        bases:
            A set of bases for each direction of the reference / 
            approximation domain.
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
            bases, 
            neglogratios[indices_debug], 
            xs[indices_debug]
        )

        return InputData(xs[indices], xs[indices_debug], fxs_debug)

    def _get_new_layer(self, xs: Tensor, neglogratios: Tensor) -> TTSIRT:
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

            xs, neglogfxs = self._eval_irt(rs)
            neglogliks = self.negloglik(xs)
            neglogpris = self.reference.eval_potential(xs)[0]

            neglogratios = self.bridge.get_ratio_func(
                self.reference,
                self.dirt_options.method,
                rs, 
                neglogliks, 
                neglogpris, 
                neglogfxs
            )

            return neglogratios

        if self.prev_approx is None:
            
            # Generate debugging and initialisation samples
            # bases_i = bases_list[min(self.n_layers, 1)] 
            input_data = self._get_inputdata(self.bases, xs, neglogratios)

            if self.n_layers == 0:
                approx = None 
                tt_data = None
            else:
                # Use previous approximation as a starting point
                approx = deepcopy(self.irts[self.n_layers-1].approx)
                tt_data = deepcopy(self.irts[self.n_layers-1].approx.tt_data)

            sirt = TTSIRT(
                updated_func,
                prior=self.prior,
                bases=self.bases.polys,
                prev_approx=approx,
                options=self.sirt_options,
                input_data=input_data,
                tt_data=tt_data,
                defensive=self.dirt_options.defensive
            )
        
        else:
            
            ind_prev = max(self.prev_approx.keys())
            sirt_prev = self.prev_approx[min(ind_prev, self.n_layers)]
            
            input_data = self._get_inputdata(
                sirt_prev.bases,
                xs, 
                neglogratios
            )

            sirt = TTSIRT(
                updated_func,
                approx=sirt_prev.approx,
                options=self.sirt_options,
                input_data=input_data, 
                defensive=self.dirt_options.defensive
            )
        
        return sirt
    
    def _build(self) -> None:
        """Constructs a DIRT to approximate a given probability 
        density.
        """

        rs = self.reference.random(self.dim, self.pre_sample_size)
        neglogliks = self.negloglik(rs)
        neglogpris = self.reference.eval_potential(rs)[0]
        
        while self.n_layers < self.dirt_options.max_layers:
            
            # Transform samples such that they are distributed 
            # according to current approximation
            xs, neglogfxs = self._eval_irt(rs)
            neglogliks = self.negloglik(xs)
            neglogpris = self.reference.eval_potential(xs)[0]
            # neglogliks, neglogpris = func(xs)
        
            self.bridge.adapt_density(
                self.dirt_options.method, 
                neglogliks, 
                neglogpris, 
                neglogfxs
            )

            neglogratios = self.bridge.get_ratio_func(
                self.reference,
                self.dirt_options.method, 
                rs, 
                neglogliks, 
                neglogpris, 
                neglogfxs
            )
            
            log_weights = self.bridge.compute_log_weights(
                neglogliks,
                neglogpris, 
                neglogfxs
            )

            if self.dirt_options.verbose:
                self.bridge.print_progress(
                    log_weights, 
                    neglogliks, 
                    neglogpris, 
                    neglogfxs
                )

            # TODO: change this to re-order by weights
            resampled_inds = self.bridge.resample(log_weights)

            self.irts[self.n_layers] = self._get_new_layer(
                xs[resampled_inds], 
                neglogratios[resampled_inds]
            )

            self.log_z += self.irts[self.n_layers].z.log()
            self.num_eval += self.irts[self.n_layers].approx.num_eval

            self.n_layers += 1
            if self.bridge.is_last:
                if self.dirt_options.verbose:
                    dirt_info("DIRT construction complete.")
                return

        msg = "Maximum number of DIRT layers reached. Building final layer..."
        warnings.warn(msg)

        raise NotImplementedError()

        xs, neglogfxs = self.eval_irt(rs)
        neglogliks, neglogpris = func(xs)
        
        log_proposal = -neglogfxs
        log_target = -neglogliks - neglogpris
        div_h2 = compute_f_divergence(log_proposal, log_target)
        div_h = div_h2.sqrt()

        msg = [f"Iter: {self.n_layers}", f"DHell: {div_h:.4f}"]
        if self.dirt_options.verbose:
            dirt_info(" | ".join(msg))
            dirt_info("DIRT construction complete.")
        return

    def eval_potential(
        self, 
        xs: Tensor,
        n_layers: Tensor = torch.inf,
        subset: str|None = None
    ) -> Tensor:
        r"""Evaluates the potential function.
        
        Returns the joint potential function, or the marginal potential 
        function for the first $k$ variables or the last $k$ variables,
        corresponding to the pullback of the reference measure under a 
        given number of layers of the DIRT.
        
        Parameters
        ----------
        xs:
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
        neglogfxs = self.eval_rt(xs, n_layers, subset)[1]
        return neglogfxs
    
    def eval_pdf(
        self, 
        xs: Tensor,
        n_layers: Tensor = torch.inf,
        subset: str|None = None
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
        fxs:
            An $n$-dimensional vector containing the value of the 
            approximation to the target density evaluated at each 
            element in `xs`.
        
        """
        neglogfxs = self.eval_potential(xs, n_layers, subset)
        fxs = torch.exp(-neglogfxs)
        return fxs

    def _eval_rt(
        self,
        xs: Tensor,
        n_layers: Tensor = torch.inf,
        subset: str|None = None
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
            
            zs = self.irts[i]._eval_rt(rs)
            neglogsirts = self.irts[i]._eval_potential(rs, subset)

            rs = self.reference.invert_cdf(zs)
            neglogrefs = self.reference.eval_potential(rs)[0]
            neglogfxs += neglogsirts - neglogrefs

        neglogrefs = self.reference.eval_potential(rs)[0]
        neglogfxs += neglogrefs

        return rs, neglogfxs
    
    def eval_rt(
        self,
        ms: Tensor,
        n_layers: Tensor = torch.inf,
        subset: str|None = None
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
            mappings evaluated at each value of `xs`.
        neglogfxs:
            An $n$-dimensional vector containing the potential function 
            of the pullback of the reference density under the current 
            composition of mappings, evaluated at each sample in `xs`.

        """

        n_layers = min(n_layers, self.n_layers)

        xs = self.prior.Q_inv(ms)

        rs, neglogfxs = self._eval_rt(xs, n_layers, subset)

        # TODO: could eventually move this to the start
        neglogabsdet_ms = self.prior.neglogabsdet_Q_inv(ms)
        neglogfxs += neglogabsdet_ms

        return rs, neglogfxs

    def _eval_irt(
        self, 
        rs: Tensor, 
        n_layers: int = torch.inf,
        subset: str|None = None
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
            xs, neglogsirts = self.irts[i]._eval_irt(zs, subset)
            neglogfxs += neglogsirts - neglogrefs

        return xs, neglogfxs

    def eval_irt(
        self, 
        ms: Tensor, 
        n_layers: int = torch.inf,
        subset: str|None = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the deep inverse Rosenblatt transport.

        Parameters
        ----------
        ms:
            An $n \times k$ matrix containing samples distributed 
            according to the prior.
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

        rs = self.prior.Q_inv(ms)
        xs, neglogfxs = self._eval_irt(rs, n_layers, subset)

        # Map samples back into actual domain
        ms = self.prior.Q(xs)
        neglogabsdet_ms = self.prior.neglogabsdet_Q_inv(ms)
        neglogfms = neglogfxs + neglogabsdet_ms

        return ms, neglogfms
    
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
        rs = self.reference.random(self.dim, n)
        xs = self.eval_irt(rs)[0]
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
        rs = self.reference.sobol(self.dim, n)
        xs = self.eval_irt(rs)[0]
        return xs