import abc
from typing import Callable, Tuple

import torch

from .abstract_irt import AbstractIRT
from .sirt import SIRT
from ..approx_bases import ApproxBases
from ..bridging_densities import Bridge, Tempering1
from ..domains import BoundedDomain
from ..input_data import InputData
from ..options import ApproxOptions, DIRTOptions, TTOptions
from ..references import Reference, GaussianReference
from ..utils import dirt_info


class DIRT(abc.ABC):

    def __init__(
        self, 
        func: Callable, 
        bases: ApproxBases, 
        bridge: Bridge|None=None,
        reference: Reference|None=None,
        sirt_options: ApproxOptions|None=None,
        dirt_options: DIRTOptions|None=None,
        init_samples: torch.Tensor|None=None,
        prev_approx=None  # TODO: fix this (set as None if not passed in) and add type annotation
    ):
        """Deep inverse Rosenblatt transform.

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
        self.bridge = bridge
        self.reference = reference 
        self.sirt_options = sirt_options
        self.dirt_options = dirt_options
        self.init_samples = init_samples
        self.prev_approx = prev_approx

        self.irts: dict[int, AbstractIRT] = {}
        self.num_eval: int = 0
        self.log_z: float = 0.0

        bases, self.dim = self.build_bases(self.bases, self.reference)
        self.build(func, bases)
        return

    @property 
    def num_layers(self) -> int:
        return self.bridge.num_layers
    
    @num_layers.setter
    def num_layers(self, value):
        self.bridge.num_layers = value 
        return

    @property
    def pre_sample_size(self) -> int:
        return self.dirt_options.num_samples + self.dirt_options.num_debugs

    @abc.abstractmethod
    def get_new_layer(
        func: Callable, 
        bases: list[ApproxBases], 
        sirt_options: ApproxOptions, 
        samples: torch.Tensor, 
        density: torch.Tensor
    ) -> SIRT:
        """TODO: write docstring."""
        return

    def initialise(
        self, 
        basis: ApproxBases
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
            domain.
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
            xs, neglogfxs = basis.sample_measure(self.pre_sample_size)
            neglogliks, neglogpris = self.func(xs)
        else:
            xs = self.init_samples
            neglogliks, neglogpris = self.bridge.eval(self.func, xs)
            neglogfxs = neglogpris  # Samples are drawn from the prior
            self.bridge = self.bridge.set_init(neglogliks)  # TODO: write this

        return xs, neglogliks, neglogpris, neglogfxs

    def get_potential_to_density(
        self, 
        bases: ApproxBases, 
        neglogratio: torch.Tensor, 
        rs: torch.Tensor  # Not sure what this is supposed to be? Samples from the reference (not [-1, 1]), although the bases.reference2domain and bases.eval_measure_potential_reference would suggest otherwise...
    ) -> torch.Tensor:
        """Returns the (square-rooted?) density we aim to approximate.
        
        TODO: finish docstring. This seems like a pretty key function..."""
        
        # TODO: figure out what's going on here. I have no idea why dxdrs is being computed.
        _, dxdrs = bases.reference2domain(rs)
        neglogws = bases.eval_measure_potential_reference(rs)

        log_ys = -0.5 * (neglogratio - neglogws - dxdrs.log().sum(dim=1))
        return torch.exp(log_ys)

    def get_inputdata(
        self, 
        bases: ApproxBases, 
        xs: torch.Tensor, 
        neglogratio: torch.Tensor 
    ) -> InputData:
        """TODO: write docstring.
        
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
            TODO: write this.
            
        """

        if self.dirt_options.num_debugs == 0:
            return InputData(xs)  # TODO: figure out if the indices are actually needed.
        
        indices = torch.arange(self.dirt_options.num_samples)
        indices_debug = torch.arange(self.dirt_options.num_debugs)
        indices_debug += self.dirt_options.num_samples

        fxs_debug = self.get_potential_to_density(
            bases, 
            neglogratio[indices_debug], 
            xs[indices_debug]
        )

        return InputData(xs[indices], xs[indices_debug], fxs_debug)

    def eval_potential(
        self, 
        xs: torch.Tensor,
        num_layers: torch.Tensor=torch.inf
    ) -> torch.Tensor:
        """Evaluates the potential function associated with the 
        pushforward of the reference measure under a given number of 
        layers of the current DIRT.
        
        Parameters
        ----------
        xs:
            An n * d matrix containing a set of samples to evaluate the 
            pushforward for.
        num_layers:
            The number of layers of the current DIRT construction to
            push forward the samples under.

        Returns
        -------
        neglogfxs:
            An n-dimensional vector containing the potential function
            of the desnity of the pushforward measure evaluated at each
            element of xs. 

        """

        num_layers = min(num_layers, self.num_layers)
        _, neglogfxs = self.eval_rt(xs, num_layers)

        return neglogfxs

    def eval_rt(
        self,
        xs: torch.Tensor,
        num_layers: torch.Tensor=torch.inf
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the deep Rosenblatt transport X = T(R), where 
        R is the reference random variable and X is the target random
        variable.
        
        TODO: tidy the below up.
        Parameters
        ----------
        xs:
            An n * d matrix of random variables drawn from the density 
            defined by the current SIRT.
        num_layers:
            The number of layers of SIRTS to push the random variables 
            forward under.

        Returns
        -------
        zs:
            xx
        neglogfxs:
            xx
        """
        
        num_layers = min(num_layers, self.num_layers)
        zs = xs.clone()

        neglogfxs = torch.zeros(zs.shape[0])

        for l in range(num_layers+1):
            
            # The first two are fine regardless of number of layers, the third appears problematic
            us = self.irts[l].eval_rt(zs)
            neglogts = self.irts[l].eval_potential(zs)

            zs = self.reference.invert_cdf(us)
            neglogds = -self.reference.log_joint_pdf(zs)[0]

            neglogfxs += neglogts - neglogds

        neglogfs = -self.reference.log_joint_pdf(zs)[0]
        neglogfxs += neglogfs # TODO: figure out what these things are

        # from matplotlib import pyplot as plt
        # plt.clf()
        # n = int(neglogfxs.numel() ** 0.5)
        # plt.pcolormesh(torch.exp(-neglogfxs).reshape(n, n))
        # plt.colorbar()
        # plt.show()

        return zs, neglogfxs

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
            An n-dimensional vector containing the pushforward of the 
            reference density under the deep inverse Rosenblatt 
            transport, evaluated at each sample in xs.

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

        for l in range(num_layers-1, -1, -1):

            # Evaluate reference density
            neglogrefs = -self.reference.log_joint_pdf(xs)[0]

            # Evaluate the current mapping Q
            zs = self.reference.eval_cdf(xs)[0]
            xs, neglogsirts = self.irts[l].eval_irt_nograd(zs)
            neglogfxs += neglogsirts - neglogrefs

        return xs, neglogfxs
    
    def build_bases(self, bases, reference: Reference):
        """TODO: need to do the cases where we have, e.g., a list of 
        approximation bases rather than just one."""

        if not isinstance(bases, ApproxBases):
            raise NotImplementedError("TODO")

        bases_list = [  # TODO: maybe e.g. a dictionary would be better here
            ApproxBases(bases.polys, bases.domains, bases.dim),
            ApproxBases(bases.polys, reference.domain, bases.dim)
        ]

        return bases_list, bases.dim

    def build(
        self,
        func: Callable, 
        bases: list[ApproxBases]
    ) -> None:
        """Constructs the deep inverse Rosenblatt transport using a
        composition of mappings.
        
        Parameters
        ----------
        func:
            A function that returns the negative log-likelihood and 
            negative log-prior density associated with a sample (or 
            samples) from the approximation domain.
        bases: 
            TODO

        Returns
        -------
        None
        
        """
        
        while self.num_layers < self.dirt_options.max_layers:

            if self.num_layers == 0:
                (xs, neglogliks, 
                 neglogpris, neglogfxs) = self.initialise(bases[0])
            else:
                # Push forward a set of samples from the reference 
                # density to the current approximation
                xs, neglogfxs = self.eval_irt(rs)
                neglogliks, neglogpris = func(xs)
        
            # Determine parameters (e.g., beta) associated with next target density
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

            # Generate a new set of particles with equal weights
            # TODO: move this to its own function
            resampled_inds = torch.multinomial(
                input=log_weights.exp(), 
                num_samples=self.pre_sample_size, 
                replacement=True
            )

            self.irts[self.num_layers] = self.get_new_layer(
                func, 
                bases, 
                self.sirt_options, 
                xs[resampled_inds], 
                neglogratio[resampled_inds]
            )

            self.log_z += self.irts[self.num_layers].z.log()
            self.num_eval += self.irts[self.num_layers].approx.num_eval

            rs = self.reference.random(self.dim, self.pre_sample_size)

            if self.bridge.is_last:
                dirt_info("DIRT construction complete.")
                return
            self.num_layers += 1

        # Finish off the last layer
        # TODO: it might be good to have a warning in here.
        xs, neglogfxs = self.eval_irt(rs)
        neglogliks, neglogpris = func(xs)

        raise NotImplementedError("TODO: finish this.")