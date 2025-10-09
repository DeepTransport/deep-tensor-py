from .lagrange_p import LagrangeP
from .piecewise_chebyshev_cdf import PiecewiseChebyshevCDF


class LagrangePCDF(LagrangeP, PiecewiseChebyshevCDF):

    def __init__(self, poly: LagrangeP, error_tol: float):
        LagrangeP.__init__(self, poly.order, poly.num_elems, poly.device)
        PiecewiseChebyshevCDF.__init__(self, poly, error_tol)
        return