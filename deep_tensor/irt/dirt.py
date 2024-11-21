from typing import Callable

import torch

from .tt_sirt import TTSIRT
from ..approx_bases import ApproxBases
from ..bridging_densities import Bridge, Tempering1
from ..domains import BoundedDomain
from ..input_data import InputData
from ..options import DIRTOptions, TTOptions
from ..polynomials import Legendre
from ..references import Reference, GaussianReference


_DEFAULT_BRIDGE = Tempering1(min_beta=1e-3, ess_tol=0.4)
_DEFAULT_REFERENCE = GaussianReference(
    mu=0, 
    sigma=1, 
    domain=BoundedDomain(bounds=torch.tensor([-4.0, 4.0]))
)

_DEFAULT_INIT_SAMPLES = []
_DEFAULT_DOMAIN = BoundedDomain() # bounds = torch.Tensor([-1, 1])
_DEFAULT_POLYNOMIAL = Legendre(30)

_DEFAULT_DIRT_OPTIONS = DIRTOptions()
_DEFAULT_SIRT_OPTIONS = TTOptions(max_als=1, tt_method="random")


class DIRT():
    """Deep inverse Rosenblatt transforms.

    Properties
    ----------
    func:
        Function that returns (quantities proportional to) the 
        negative log-likelihood and negative log-prior density
        associated with a given set of parameters.
    bases:
        The bases for ...
    bridge:
        An object used to construct successive approximations to the 
        target distribution.
    reference:
        The reference distribution.
    sirt_options:
        ...
    dirt_options:
        ...
    init_samples: 
        Samples to initialise ...  (drawn from the prior).
    prev_approx:
        ...?
    
    """

    def __init__(
        self, 
        func: Callable, 
        bases: ApproxBases, 
        bridge: Bridge=_DEFAULT_BRIDGE,
        reference: Reference=_DEFAULT_REFERENCE,
        sirt_options: TTOptions=_DEFAULT_SIRT_OPTIONS,
        dirt_options: DIRTOptions=_DEFAULT_DIRT_OPTIONS,
        init_samples=_DEFAULT_INIT_SAMPLES,  # TODO: add type annotation
        prev_approx={}
    ):
        
        self.func = func
        self.bases = bases
        self.bridge = bridge
        self.reference = reference 

        self.sirt_options = sirt_options
        self.dirt_options = dirt_options
        
        self.init_samples = init_samples
        self.prev_approx = prev_approx

        self.irts: dict[TTSIRT] = {}
        self.num_evals: int = 0
        self.logz: float = 0.0

        bases, self.dim = self.build_bases(bases, reference)
        self.build(func, bases, sirt_options)

    @property
    def pre_sample_size(self) -> int:
        return self.dirt_options.num_samples + self.dirt_options.num_debugs

    def eval_irt(
        self, 
        z: torch.Tensor, 
        num_layers: int|None=None
    ):
        """Evaluates the deep Rosenblatt transport X = T(Z), where Z is
        the reference random variable and X is the target random 
        variable.

        Parameters
        ----------
        z:
            Reference random variable.
        num_layers: 
            Number of layers.

        TODO: this has a few different cases / output arguments. Need 
        to figure out how to split it up. This function is currently
        for the case where nargout=2.

        """

        if num_layers is not None:
            num_layers = min(num_layers, self.bridge.num_layers)
        else: 
            num_layers = self.bridge.num_layers

        x = z 

        neglogfx = -self.reference.log_joint_pdf(x)[0]

        for l in range(num_layers-1, 0, -1):

            # Evaluate the diagonal transform
            logd = self.reference.log_joint_pdf(x)
            u, _ = self.reference.eval_cdf(x)
            x, neglogt = self.irts[l].eval_irt(u) # TODO: write this

            # Update density
            neglogf = neglogf + neglogt + logd

        return x, neglogfx
    
    def get_potential_to_density(
        self, 
        bases: ApproxBases, 
        y: torch.Tensor, 
        z: torch.Tensor
    ) -> torch.Tensor:
        
        _, dxdz = bases.reference2domain(z)
        neglogref = bases.eval_measure_potential_reference(z)

        y = torch.exp(-0.5*(y-neglogref-torch.sum(torch.log(dxdz), 1)))
        return y # TODO: check that the dimension of the sum is correct.
        
    #     function y = potential_to_density(base, potential_fun, z)
    #         [x, dxdz] = reference2domain(base, z);
    #         y = feval(potential_fun, x);
    #         % log density of the reference measure
    #         mlogw = eval_measure_potential_reference(base, z);
    #         %
    #         % y is the potential function, use change of coordinates
    #         % y = exp( - 0.5*y + 0.5*sum(log(base.dxdz)) + 0.5*mlogw );
    #         y = exp( - 0.5*(y-mlogw-sum(log(dxdz),1)) );
    #     end

    def get_inputdata(
        self, 
        base: ApproxBases, 
        samples: torch.Tensor, 
        density: torch.Tensor 
    ) -> InputData:
            
        indices = torch.arange(self.dirt_options.num_samples)

        if self.dirt_options.num_debugs == 0:
            return InputData(samples[indices])

        indices_debug = torch.arange(self.dirt_options.num_debugs)
        indices_debug += self.dirt_options.num_samples

        tmp = self.get_potential_to_density(
            base, 
            density[indices_debug], 
            samples[indices_debug]
        )

        return InputData(samples[indices], samples[indices_debug], tmp)

    def get_new_layer(
        self, 
        func: Callable, 
        bases: list[ApproxBases], 
        sirt_options, 
        num_layers: int,
        samples, 
        density
    ):
        """Func is the one that returns neglogpdf/negloglik."""

        def updated_func(zs: torch.Tensor) -> torch.Tensor:

            return self.bridge.ratio_func(
                func, 
                zs, 
                self.eval_irt, 
                self.reference, 
                self.dirt_options.method
            )

        # updated_func = lambda z: self.bridge.ratio_func(func, self, z)

        if self.prev_approx == {}:
            if num_layers == 0:  # start from fresh(??)
                
                # Compute debugging and initialisation samples
                input_data = self.get_inputdata(
                    bases[num_layers+1], 
                    samples, 
                    density
                )

                irt = TTSIRT(
                    updated_func, 
                    bases[num_layers+1], 
                    options=sirt_options, 
                    input_data=input_data,
                    defensive=self.dirt_options.defensive
                )

            else:
                raise NotImplementedError()


    def build_bases(self, bases, reference: Reference):
        # TODO: need to do the cases where we have, e.g., a list of 
        # approximate bases rather than just one.

        """if isa(arg, 'cell')
        % This should contain 2 cells: for levels 0 and 1
        if (numel(arg)>2)
            warning('bases cells 3:%d are not used and will be ignored', numel(arg));
            bases = arg(1:2);
        elseif (numel(arg)<2)
            warning('repeat the first base');
            bases = repmat(arg(1), 1, 2);
        else
            bases = arg;
        end
        for i = 1:2
            if (isa(bases{i}, 'ApproxBases'))
                oneds = bases{i}.oneds;
                if i == 1
                    doms = bases{i}.oned_domains;
                else
                    doms = ref_dom;
                end
                d = ndims(bases{i});
            else
                error('bases cells element should be ApproxBases');
            end
            bases{i} = ApproxBases(oneds,doms,d);
        end"""

        if not isinstance(bases, ApproxBases):
            raise NotImplementedError()

        # Form a list containing the ...?
        bases_list = [
            ApproxBases(bases.polys, bases.in_domains, bases.dim),
            ApproxBases(bases.polys, [reference.domain], bases.dim)
        ]

        return bases_list, bases.dim

    def build(
        self,
        func: Callable, 
        bases: list[ApproxBases], 
        sirt_options: TTOptions
    ):

        # neglogfx = density the particles are sampled from??
        
        while self.bridge.num_layers < self.dirt_options.max_layers:

            if self.bridge.num_layers == 0:

                # Some initialisation stuff (this could probably be moved to a separate function)
                if self.init_samples == []:
                    samples, neglogfx = bases[0].sample_measure(self.pre_sample_size)
                    neglogliks, neglogpris = self.bridge.eval(func, samples)
                else:
                    samples = self.init_samples
                    neglogliks, neglogpris = self.bridge.eval(func, samples)
                    neglogfx = neglogpris
                    self.bridge = self.bridge.set_init(neglogliks) # TODO: write this function.

            else:
                # reuse samples
                x, neglogfx = self.eval_irt(samples)
                neglogliks, neglogpris = self.bridge.eval(func, x)
        
            # Generate new target density
            self.bridge.adapt_density(
                self.dirt_options.method, 
                neglogliks, 
                neglogpris, 
                neglogfx
            )

            density = self.bridge.get_ratio_func(
                self.reference,
                self.dirt_options.method, 
                samples, 
                neglogliks, 
                neglogpris, 
                neglogfx
            )
            
            log_weights = self.bridge.print_progress(
                neglogliks, 
                neglogpris, 
                neglogfx
            )

            # Generate a new set of particles with equal weights
            resampled_inds = torch.multinomial(
                input=torch.exp(log_weights), 
                num_samples=self.pre_sample_size, 
                replacement=True
            )

            self.irts[self.bridge.num_layers] = self.get_new_layer(
                func, 
                bases, 
                sirt_options, 
                self.bridge.num_layers, 
                samples[resampled_inds], 
                density[resampled_inds]
            )

            self.bridge.num_layers += 1
        # %
        # obj.logz = obj.logz + log(obj.irts{n_layers+1}.z);
        # obj.n_eval = obj.n_eval + obj.irts{n_layers+1}.approx.n_eval;
        # %
        # % We need reference samples already here, in case we quit after 1 layer
        # samples = random(obj.ref, obj.d, ns);
        # % stop
        # if islast(obj.bridge)
        #     break;
        # end
        # n_layers = num_layers(obj.bridge);

class TTDIRT(DIRT):

    def __init__(self, func: Callable, bases: ApproxBases, **kwargs):
        super().__init__(func, bases, **kwargs)