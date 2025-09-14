__version__ = "0.0.4"

import torch
torch.set_default_dtype(torch.float64)

from .bridging_densities import SingleLayer, Tempering, SigmoidSmoothing
from .debiasing.importance_sampling import (
    ImportanceSamplingResult, 
    run_importance_sampling
)
from .debiasing.mcmc import (
    MCMCResult, 
    run_irt_pcn, 
    run_cirt_pcn,
    run_independence_sampler
)
from .debiasing.mcmc_new import pCNKernel, MCMC
from .domains import (
    AlgebraicMapping, 
    BoundedDomain, 
    LinearDomain, 
    LogarithmicMapping
)
from .ftt import ApproxBases, Direction, FTT, EFTT, TT, TTOptions
from .irt import DIRT, DIRTMapping, DIRTOptions, SIRT
from .polynomials import (
    Basis1D,
    Chebyshev1st, 
    Chebyshev1stTrigoCDF,
    Chebyshev2nd,
    Chebyshev2ndTrigoCDF,
    Fourier,
    Hermite,
    Lagrange1, 
    Lagrange1CDF,
    LagrangeP,
    Laguerre, 
    Legendre,
    Piecewise,
    PiecewiseCDF,
    Spectral,
    construct_cdf
)
from .preconditioners import (
    GaussianMapping,
    IdentityMapping,
    Preconditioner, 
    UniformMapping
)
from .references import Reference, GaussianReference, UniformReference
from .target_functions import RareEventFunc, TargetFunc
from .tools import compute_f_divergence