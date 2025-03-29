from .basis_1d import Basis1D
from .cdf_1d import CDF1D

from .piecewise.lagrange_1 import Lagrange1
from .piecewise.lagrange_1_cdf import Lagrange1CDF
from .piecewise.lagrange_p import LagrangeP
from .piecewise.lagrange_p_cdf import LagrangePCDF

from .spectral.chebyshev_2nd_cdf import Chebyshev2ndCDF
from .spectral.chebyshev_1st import Chebyshev1st
from .spectral.chebyshev_1st_cdf import Chebyshev1stCDF
from .spectral.chebyshev_1st_trigo_cdf import Chebyshev1stTrigoCDF
from .spectral.chebyshev_2nd import Chebyshev2nd
from .spectral.chebyshev_2nd_trigo_cdf import Chebyshev2ndTrigoCDF
from .spectral.fourier import Fourier
from .spectral.fourier_cdf import FourierCDF
from .spectral.hermite import Hermite
from .spectral.hermite_cdf import HermiteCDF
from .spectral.laguerre import Laguerre
from .spectral.laguerre_cdf import LaguerreCDF
from .spectral.legendre import Legendre


POLY_CDFS = {
    Chebyshev1st: Chebyshev1stTrigoCDF,
    Chebyshev2nd: Chebyshev2ndTrigoCDF,
    Fourier: FourierCDF,
    Hermite: HermiteCDF,
    Lagrange1: Lagrange1CDF,
    LagrangeP: LagrangePCDF,
    Laguerre: LaguerreCDF,
    Legendre: Chebyshev2ndCDF
}


def construct_cdf(poly: Basis1D, **kwargs: dict) -> CDF1D:
    """Selects the one-dimensional CDF function for a given polynomial
    basis.
    """

    try: 
        return POLY_CDFS[type(poly)](poly, **kwargs)
    except KeyError:
        msg = f"CDF not implemented for polynomial of type {type(poly)}."
        raise Exception(msg)