import torch

from .bridging_densities import SingleLayer, Tempering
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
    GaussianMapping,
    IdentityMapping,
    Preconditioner, 
    UniformMapping
)
from .references import Reference, GaussianReference, UniformReference
from .tools import compute_f_divergence