# TODO: finish this

import torch
torch.set_default_dtype(torch.float64)

from .approx_bases import ApproxBases
from .bridging_densities import Tempering1
from .directions import Direction
from .domains import BoundedDomain
from .input_data import InputData
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
from .irt import TTDIRT, TTSIRT