from .cubic_hermite import CubicHermite
from .piecewise_chebyshev_cdf import PiecewiseChebyshevCDF


class CubicHermiteCDF(CubicHermite, PiecewiseChebyshevCDF):

    def __init__(self, poly: CubicHermite, **kwargs):
        CubicHermite.__init__(self, poly.num_elems)
        PiecewiseChebyshevCDF.__init__(self, poly, **kwargs)
        return