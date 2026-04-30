__version__ = "1.2.0"

import torch
torch.set_default_dtype(torch.float64)

from .bridging_densities import SigmoidSmoothing, GaussianSmoothing, SingleLayer, Tempering
from .debiasing.importance_sampling import (
    ImportanceSamplingResult, 
    run_importance_sampling
)
from .debiasing.mcmc import (
    MCMC,
    MCMCResult, 
    run_independence_sampler,
    pCNKernel
)
from .debiasing.stats import estimate_iact
from .domains import BoundedDomain, LinearDomain
from .ftt import Direction, FTT, EFTT, EFTTOptions, TT, TTOptions
from .irt import DIRT, DIRTMapping, DIRTOptions, SIRT
from .polynomials import (
    Basis1D,
    Chebyshev1st, 
    Chebyshev1stTrigoCDF,
    Chebyshev2nd,
    Chebyshev2ndTrigoCDF,
    Fourier,
    Lagrange1, 
    Lagrange1CDF,
    LagrangeP,
    Legendre,
    Piecewise,
    PiecewiseCDF,
    Spectral,
    construct_cdf
)
from .preconditioners import (
    AffineMapping,
    GaussianMapping,
    IdentityMapping,
    Preconditioner, 
    UniformMapping
)
from .references import Reference, GaussianReference, UniformReference
from .subspaces import FixedSubspace, IdentitySubspace, LikelihoodInformedSubspace
from .target_functions import RareEventFunc, TargetFunc
from .tools import compute_f_divergence