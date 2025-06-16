import torch
torch.set_default_dtype(torch.float64)

from .bridging_densities import SingleLayer, Tempering
from .debiasing import (
    ImportanceSamplingResult,
    MCMCResult,
    run_dirt_pcn, 
    run_importance_sampling, 
    run_independence_sampler
)
from .domains import (
    AlgebraicMapping, 
    BoundedDomain, 
    LinearDomain, 
    LogarithmicMapping
)
from .ftt import ApproxBases, Direction, InputData, TTData, TTFunc
from .irt import DIRT, SIRT, SavedDIRT
from .options import TTOptions, DIRTOptions
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
    Preconditioner, 
    IdentityMapping,
    UniformMapping
)
from .references import Reference, GaussianReference, UniformReference
from .tools import estimate_ess_ratio, compute_f_divergence