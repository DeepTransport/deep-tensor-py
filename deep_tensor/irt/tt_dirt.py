from copy import deepcopy
from typing import Callable, Tuple
import warnings

import torch

from .tt_sirt import TTSIRT
from ..ftt import ApproxBases, InputData
from ..bridging_densities import Bridge, Tempering1
from ..domains import BoundedDomain
from ..options import DIRTOptions, TTOptions
from ..references import Reference, GaussianReference
from ..tools import compute_f_divergence
from ..tools.printing import dirt_info


class TTDIRT():

    def __init__(
        self, 
        func: Callable, 
        bases: ApproxBases, 
        bridge: Bridge|None=None,
        reference: Reference|None=None,
        sirt_options: TTOptions|None=None,
        dirt_options: DIRTOptions|None=None,
        init_samples: torch.Tensor|None=None,
        prev_approx=None  # TODO: fix this (set as None if not passed in) and add type annotation
    ):
        """Class that implements the deep inverse Rosenblatt transport.

        Properties
        ----------
        func:
            Function that returns (quantities proportional to) the 
            negative log-likelihood and negative log-prior density
            associated with a given set of parameters.
        bases:
            The set of polynomial bases associated with each dimension
            of the approximation domain.
        bridge:
            An object used to construct successive approximations to the 
            target distribution.
        reference:
            The reference distribution.
        sirt_options:
            Options for constructing the SIRT approximation to the 
            bridging density at each iteration.
        dirt_options:
            Options for constructing the DIRT approximation to the 
            target density.
        init_samples: 
            A set of samples, drawn from the prior, to intialise the 
            FTT with. If these are not passed in, a set of samples will 
            instead be drawn from the reference measure and transformed 
            into the approximation domain.
        prev_approx:
            ...?
        
        """

        if bridge is None:
            bridge = Tempering1(min_beta=1e-3, ess_tol=0.4)
        
        if reference is None:
            bounds = torch.tensor([-4.0, 4.0])
            domain = BoundedDomain(bounds=bounds)
            reference = GaussianReference(mu=0.0, sigma=1.0, domain=domain)
        
        if sirt_options is None:
            sirt_options = TTOptions(max_als=1, tt_method="random")
        
        if dirt_options is None:
            dirt_options = DIRTOptions()

        self.func = func
        self.bases = bases
        self.dim = bases.dim
        self.bridge = bridge
        self.reference = reference 
        self.sirt_options = sirt_options
        self.dirt_options = dirt_options
        self.init_samples = init_samples
        self.prev_approx = prev_approx
        self.pre_sample_size = (self.dirt_options.num_samples 
                                + self.dirt_options.num_debugs)

        if self.init_samples is not None:
            if self.init_samples.shape[0] < self.pre_sample_size:
                msg = ("More initialisation samples are required. "
                       + f"Need {self.pre_sample_size}, "
                       + f"got {self.init_samples.shape[0]}.")
                raise Exception(msg)

        self.irts: dict[int, TTSIRT] = {}
        self.num_eval: int = 0
        self.log_z: float = 0.0

        bases_list = self.build_bases(self.bases, self.reference)
        self.build(func, bases_list)
        return

    @property 
    def num_layers(self) -> int:
        return self.bridge.num_layers
    
    @num_layers.setter
    def num_layers(self, value):
        self.bridge.num_layers = value 
        return

    def build_bases(
        self, 
        bases: ApproxBases, 
        reference: Reference
    ) -> list[ApproxBases]:
        """Returns a list of bases for the first and second levels of 
        DIRT construction.

        Parameters
        ----------
        bases:
            An ApproxBases object for the approximation domain.
        reference:
            The reference density.

        Returns
        -------
        bases_list:
            A list of the bases for the first and second levels of DIRT
            construction.
                
        """

        if not isinstance(bases, ApproxBases):
            msg = ("Currently, only a set of ApproxBases can be passed "
                   + "into 'build_bases()'.")
            raise NotImplementedError(msg)

        bases_list = [
            ApproxBases(bases.polys, bases.domains, bases.dim),
            ApproxBases(bases.polys, reference.domain, bases.dim)
        ]

        return bases_list

    def initialise(
        self, 
        bases: ApproxBases
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates a set of samples to initialise the FTT with.
        
        Parameters
        ----------
        bases: 
            A set of bases for the approximation domain.

        Returns
        -------
        xs:
            An n * d matrix containing samples from the approximation 
            domain. If init_samples (samples drawn from the prior) are 
            available, these are used; otherwise, a set of samples
            are drawn from the weighting function associated with the 
            bases for the reference density.     
        neglogliks:
            An n-dimensional vector containing the negative of the
            log-likelihood function evaluated at each sample.
        neglogpris: 
            An n-dimensional vector containing the negative logarithm
            of the prior density evaluated at each sample.
        neglogfxs:
            An n-dimensional vector containing the negative logarithm 
            of the density the samples are drawn from.
        
        """

        if self.init_samples is None:
            # When piecewise polynomials are used, measure is uniform
            xs, neglogfxs = bases.sample_measure(self.pre_sample_size)
            neglogliks, neglogpris = self.func(xs)
        else:
            xs = self.init_samples
            neglogliks, neglogpris = self.func(xs)
            neglogfxs = neglogpris  # Samples are drawn from the prior
            self.bridge.set_init(neglogliks)

        return xs, neglogliks, neglogpris, neglogfxs

    def get_potential_to_density(
        self, 
        bases: ApproxBases, 
        neglogratios: torch.Tensor, 
        rs: torch.Tensor
    ) -> torch.Tensor:
        """Returns the (square-rooted?) density we aim to approximate.
        """
        neglogwrs = bases.eval_measure_potential(rs)[0]
        log_ys = -0.5 * (neglogratios - neglogwrs)
        return torch.exp(log_ys)

    def get_inputdata(
        self, 
        bases: ApproxBases, 
        xs: torch.Tensor, 
        neglogratio: torch.Tensor 
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
        neglogratio:
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
        indices_debug = torch.arange(self.dirt_options.num_debugs)
        indices_debug += self.dirt_options.num_samples

        fxs_debug = self.get_potential_to_density(
            bases, 
            neglogratio[indices_debug], 
            xs[indices_debug]
        )

        return InputData(xs[indices], xs[indices_debug], fxs_debug)

    def get_new_layer(
        self, 
        func: Callable, 
        bases: list[ApproxBases], 
        sirt_options: TTOptions, 
        xs: torch.Tensor, 
        neglogratios: torch.Tensor
    ) -> TTSIRT:
        """Constructs a new SIRT to add to the current composition of 
        SIRTs.

        Parameters
        ----------
        func:
            Function that returns the negative log-likelihood and 
            negative log-prior density of a sample.
        bases_list:
            A list of the bases used when constructing the first two 
            levels of DIRT.
        sirt_options:
            Options used when constructing the SIRT associated with the 
            layer.
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

        def updated_func(rs: torch.Tensor) -> torch.Tensor:

            return self.bridge.ratio_func(
                func, 
                rs, 
                self.eval_irt, 
                self.reference, 
                self.dirt_options.method
            )

        if self.prev_approx is None:
            
            # Generate debugging and initialisation samples
            bases_i = bases[min(self.num_layers, 1)] 
            input_data = self.get_inputdata(bases_i, xs, neglogratios)

            # Start from fresh (TODO: figure out why this happens on 
            # the second iteration?)
            if self.num_layers <= 1:
                approx = None 
                tt_data = None
            else:
                approx = deepcopy(self.irts[self.num_layers-1].approx)
                tt_data = deepcopy(self.irts[self.num_layers-1].approx.data)

            sirt = TTSIRT(
                updated_func,
                bases=bases_i,
                approx=approx,
                options=sirt_options,
                input_data=input_data,
                tt_data=tt_data,
                tau=self.dirt_options.defensive
            )
        
        else:
            raise NotImplementedError()
        
        return sirt

    def eval_potential(
        self, 
        rs: torch.Tensor,
        num_layers: torch.Tensor=torch.inf
    ) -> torch.Tensor:
        """Evaluates the potential function associated with the 
        pushforward of the reference measure under a given number of 
        layers of the current DIRT.
        
        Parameters
        ----------
        rs:
            An n * d matrix containing a set of samples to evaluate the 
            pushforward for.
        num_layers:
            The number of layers of the current DIRT construction to
            push forward the samples under.

        Returns
        -------
        neglogfxs:
            An n-dimensional vector containing the potential function
            of the density of the pushforward measure evaluated at each
            element of xs. 

        """
        num_layers = min(num_layers, self.num_layers)
        neglogfxs = self.eval_rt(rs, num_layers)[1]
        return neglogfxs

    def eval_rt(
        self,
        xs: torch.Tensor,
        num_layers: torch.Tensor=torch.inf
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the deep Rosenblatt transport X = T(R), where 
        R is the reference random variable and X is the target random
        variable.
        
        Parameters
        ----------
        xs:
            An n * d matrix of random variables drawn from the density 
            defined by the current DIRT.
        num_layers:
            The number of layers of SIRTS to push the random variables 
            forward under.

        Returns
        -------
        rs:
            An n * d matrix containing the composition of mappings 
            evaluated at each value of xs.
        neglogfxs:
            An n-dimensional vector containing the negative log of the
            pushforward of the reference density under the current 
            composition of mappings, evaluated at each sample in xs.

        """
        
        num_layers = min(num_layers, self.num_layers)
        rs = xs.clone()

        neglogfxs = torch.zeros(rs.shape[0])

        for i in range(num_layers):
            
            zs = self.irts[i].eval_rt(rs)
            neglogsirts = self.irts[i].eval_potential(rs)

            rs = self.reference.invert_cdf(zs)
            neglogrefs = -self.reference.log_joint_pdf(rs)[0]
            neglogfxs += neglogsirts - neglogrefs

        neglogrefs = -self.reference.log_joint_pdf(rs)[0]
        neglogfxs += neglogrefs

        return rs, neglogfxs

    def eval_irt(
        self, 
        rs: torch.Tensor, 
        num_layers: int=torch.inf
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the deep inverse Rosenblatt transport X = T(R), 
        where R is the reference random variable and X is the target 
        random variable.

        Parameters
        ----------
        rs:
            An n * d matrix containing samples distributed according to
            the reference density.
        num_layers: 
            The number of layers of the deep inverse Rosenblatt 
            transport.

        Returns
        -------
        xs: 
            An n * d matrix containing the corresponding samples from 
            the approximation domain, after applying the deep inverse 
            Rosenblatt transport.
        neglogfxs:
            An n-dimensional vector containing the negative log of the
            pushforward of the reference density under the current 
            composition of mappings, evaluated at each sample in xs.

        References
        ----------
        Cui and Dolgov (2022). Deep composition of tensor-trains using 
        squared inverse Rosenblatt transports. Eq. (48).
        Cui, Dolgov and Scheichl (2024). Deep importance sampling using 
        tensor trains with application to a-priori and a-posteriori
        rare events. Eq. (3.9).

        """

        num_layers = min(num_layers, self.num_layers)
        xs = rs.clone()

        neglogfxs = -self.reference.log_joint_pdf(xs)[0]

        for i in range(num_layers-1, -1, -1):

            # Evaluate reference density
            neglogrefs = -self.reference.log_joint_pdf(xs)[0]

            # Evaluate the current mapping Q
            zs = self.reference.eval_cdf(xs)[0]
            xs, neglogsirts = self.irts[i].eval_irt(zs)
            neglogfxs += neglogsirts - neglogrefs

        return xs, neglogfxs

    def build(
        self,
        func: Callable, 
        bases_list: list[ApproxBases]
    ) -> None:
        """Constructs a DIRT to approximate a given probability 
        density.
        
        Parameters
        ----------
        func:
            A function that returns the negative log-likelihood and 
            negative log-prior density associated with a sample (or 
            samples) from the approximation domain.
        bases_list: 
            A list of approximation bases for the first and second
            levels of DIRT construction.

        Returns
        -------
        None
        
        """
        
        while self.num_layers < self.dirt_options.max_layers:

            if self.num_layers == 0:
                (xs, neglogliks, 
                 neglogpris, neglogfxs) = self.initialise(bases_list[0])
            else:
                # Push forward a set of samples from the reference 
                # density to the current approximation
                xs, neglogfxs = self.eval_irt(rs)
                neglogliks, neglogpris = func(xs)
        
            self.bridge.adapt_density(
                self.dirt_options.method, 
                neglogliks, 
                neglogpris, 
                neglogfxs
            )

            neglogratio = self.bridge.get_ratio_func(
                self.reference,
                self.dirt_options.method, 
                xs, 
                neglogliks, 
                neglogpris, 
                neglogfxs
            )
            
            log_weights = self.bridge.compute_log_weights(
                neglogliks,
                neglogpris, 
                neglogfxs
            )

            self.bridge.print_progress(
                log_weights, 
                neglogliks, 
                neglogpris, 
                neglogfxs
            )

            resampled_inds = self.bridge.resample(log_weights)

            self.irts[self.num_layers] = self.get_new_layer(
                func, 
                bases_list, 
                self.sirt_options, 
                xs[resampled_inds], 
                neglogratio[resampled_inds]
            )

            self.log_z += self.irts[self.num_layers].z.log()
            self.num_eval += self.irts[self.num_layers].approx.num_eval
            rs = self.reference.random(self.dim, self.pre_sample_size)

            self.num_layers += 1
            if self.bridge.is_last:
                dirt_info("DIRT construction complete.")
                return

        warnings.warn("Maximum number of DIRT layers reached.")

        xs, neglogfxs = self.eval_irt(rs)
        neglogliks, neglogpris = func(xs)
        
        log_proposal = -neglogfxs
        log_target = -neglogliks - neglogpris
        div_h2 = compute_f_divergence(log_proposal, log_target)[1]

        msg = [
            f"Iter: {self.num_layers}", 
            f"DHell: {div_h2.sqrt()[0]:.4f}"
        ]
        dirt_info(msg)
        dirt_info("DIRT construction complete.")
        return