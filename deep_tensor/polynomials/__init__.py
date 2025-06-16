from .basis_1d import Basis1D
from .cdf_1d import CDF1D
from .cdf_constructor import construct_cdf

from .piecewise.piecewise import Piecewise
from .piecewise.piecewise_cdf import PiecewiseCDF
from .piecewise.lagrange_1 import Lagrange1
from .piecewise.lagrange_1_cdf import Lagrange1CDF
from .piecewise.lagrange_p import LagrangeP

from .spectral.spectral import Spectral
from .spectral.chebyshev_1st import Chebyshev1st
from .spectral.chebyshev_1st_cdf import Chebyshev1stCDF
from .spectral.chebyshev_1st_trigo_cdf import Chebyshev1stTrigoCDF
from .spectral.chebyshev_2nd import Chebyshev2nd
from .spectral.chebyshev_2nd_cdf import Chebyshev2ndCDF
from .spectral.chebyshev_2nd_trigo_cdf import Chebyshev2ndTrigoCDF
from .spectral.fourier import Fourier
from .spectral.fourier_cdf import FourierCDF
from .spectral.hermite import Hermite
from .spectral.laguerre import Laguerre
from .spectral.recurr import Recurr
from .spectral.legendre import Legendre

POLY2NAME = {
    Piecewise: "Piecewise",
    Lagrange1: "Lagrange1",
    LagrangeP: "LagrangeP",
    Chebyshev1st: "Chebyshev1st",
    Chebyshev2nd: "Chebyshev2nd",
    Fourier: "Fourier",
    Hermite: "Hermite",
    Laguerre: "Laguerre",
    Legendre: "Legendre"
}

NAME2POLY = {v: k for k, v in POLY2NAME.items()}