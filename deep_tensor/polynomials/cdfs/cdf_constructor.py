from .bounded_poly_cdf import BoundedPolyCDF
from .fourier_cdf import FourierCDF
from .lagrange_1_cdf import Lagrange1CDF
from .lagrange_p_cdf import LagrangePCDF

from ..polynomials.basis_1d import Basis1D

from ..polynomials.fourier import Fourier
from ..polynomials.lagrange_1 import Lagrange1
from ..polynomials.lagrange_p import LagrangeP
from ..polynomials.legendre import Legendre


POLY_CDFS = {
    Fourier: FourierCDF,
    Lagrange1: Lagrange1CDF,
    LagrangeP: LagrangePCDF,
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