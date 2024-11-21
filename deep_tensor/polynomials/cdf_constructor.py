from .basis_1d import Basis1D

from .lagrange_1 import Lagrange1
from .legendre import Legendre

from .bounded_poly_cdf import BoundedPolyCDF
from .lagrange_1_cdf import Lagrange1CDF


POLY_CDFS = {  # TODO: finish this
    Lagrange1: Lagrange1CDF,
    Legendre: BoundedPolyCDF
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