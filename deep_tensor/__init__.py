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
from .domains import BoundedDomain
from .ftt import FTT, EFTT, EFTTOptions, TT, TTOptions
from .irt import DIRT, DIRTMapping, DIRTOptions
from .polynomials import (
    Chebyshev1st,
    Chebyshev2nd,
    Fourier,
    Lagrange1,
    LagrangeP,
    Legendre
)
from .preconditioners import (
    GaussianMapping,
    IdentityMapping,
    Preconditioner,
    UniformMapping
)
from .references import Reference, GaussianReference, UniformReference
from .subspaces import FixedSubspace, FullSpace, LikelihoodInformedSubspace
from .target_functions import RareEventFunc, TargetFunc
from .tools import estimate_dhell

# The public API of the package. Everything else (including anything
# imported directly from a submodule) is an implementation detail.
__all__ = [
    "BoundedDomain",
    "Chebyshev1st",
    "Chebyshev2nd",
    "DIRT",
    "DIRTMapping",
    "DIRTOptions",
    "EFTT",
    "EFTTOptions",
    "FTT",
    "FixedSubspace",
    "Fourier",
    "FullSpace",
    "GaussianMapping",
    "GaussianReference",
    "GaussianSmoothing",
    "IdentityMapping",
    "ImportanceSamplingResult",
    "Lagrange1",
    "LagrangeP",
    "Legendre",
    "LikelihoodInformedSubspace",
    "MCMC",
    "MCMCResult",
    "Preconditioner",
    "RareEventFunc",
    "Reference",
    "SigmoidSmoothing",
    "SingleLayer",
    "TT",
    "TTOptions",
    "TargetFunc",
    "Tempering",
    "UniformMapping",
    "UniformReference",
    "estimate_dhell",
    "estimate_iact",
    "pCNKernel",
    "run_importance_sampling",
    "run_independence_sampler",
]
