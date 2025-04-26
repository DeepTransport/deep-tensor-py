import torch
torch.set_default_dtype(torch.float64)

from .bridging_densities import SingleLayer, Tempering
from .domains import (
    AlgebraicMapping, 
    BoundedDomain, 
    LinearDomain, 
    LogarithmicMapping
)
from .ftt import ApproxBases, Direction, InputData, TTData, TTFunc
from .irt import AbstractIRT, TTDIRT, TTSIRT
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
from .prior_transformation import PriorTransformation
from .references import GaussianReference, UniformReference
from .tools import run_mcmc