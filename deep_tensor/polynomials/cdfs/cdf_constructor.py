from .bounded_poly_cdf import BoundedPolyCDF
from .chebyshev_1st_cdf import Chebyshev1stCDF
from .chebyshev_1st_trigo_cdf import Chebyshev1stTrigoCDF
from .chebyshev_2nd_trigo_cdf import Chebyshev2ndTrigoCDF
from .hermite_cdf import HermiteCDF
from .fourier_cdf import FourierCDF
from .lagrange_1_cdf import Lagrange1CDF
from .lagrange_p_cdf import LagrangePCDF
from .laguerre_cdf import LaguerreCDF

from ..polynomials.basis_1d import Basis1D

from ..polynomials.chebyshev_1st import Chebyshev1st
from ..polynomials.chebyshev_2nd import Chebyshev2nd
from ..polynomials.hermite import Hermite
from ..polynomials.fourier import Fourier
from ..polynomials.lagrange_1 import Lagrange1
from ..polynomials.lagrange_p import LagrangeP
from ..polynomials.laguerre import Laguerre
from ..polynomials.legendre import Legendre


POLY_CDFS = {
    Chebyshev1st: Chebyshev1stTrigoCDF,
    Chebyshev2nd: Chebyshev2ndTrigoCDF,
    Fourier: FourierCDF,
    Hermite: HermiteCDF,
    Lagrange1: Lagrange1CDF,
    LagrangeP: LagrangePCDF,
    Laguerre: LaguerreCDF,
    Legendre: BoundedPolyCDF,
}


def construct_cdf(poly: Basis1D, **kwargs: dict):
    """Selects the one-dimensional CDF function for a given collocation
    basis.
    """

    try: 
        return POLY_CDFS[type(poly)](poly, **kwargs)
    except KeyError:
        msg = f"CDF not implemented for polynomial of type {type(poly)}."
        raise Exception(msg)