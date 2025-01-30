import torch
torch.set_default_dtype(torch.float64)

from .bridging_densities import Tempering1
from .domains import BoundedDomain
from .ftt import ApproxBases, Direction, InputData, TTFunc
from .irt import TTDIRT, TTSIRT
from .options import TTOptions
from .polynomials import (
    Basis1D,
    Chebyshev1st, 
    Chebyshev2ndUnweighted,
    Fourier,
    Lagrange1, 
    Lagrange1CDF,
    LagrangeP, 
    Legendre,
    Piecewise,
    PiecewiseCDF,
    Spectral
)