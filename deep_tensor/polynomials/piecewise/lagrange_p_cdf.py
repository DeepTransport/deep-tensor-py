from .lagrange_p import LagrangeP
from .piecewise_chebyshev_cdf import PiecewiseChebyshevCDF


class LagrangePCDF(LagrangeP, PiecewiseChebyshevCDF):

    def __init__(self, poly: LagrangeP, **kwargs):
        LagrangeP.__init__(self, poly.order, poly.num_elems)
        PiecewiseChebyshevCDF.__init__(self, poly, **kwargs)
        return