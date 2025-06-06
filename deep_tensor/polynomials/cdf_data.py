import abc
import dataclasses

from torch import Tensor


@dataclasses.dataclass
class CDFData(abc.ABC):
    n_cdfs: int
    poly_coef: Tensor
    cdf_poly_grid: Tensor
    poly_norm: Tensor
    

@dataclasses.dataclass
class CDFDataLagrange1(CDFData):
    """Class containing information on a single CDF, or set of CDFs, 
    for a Lagrange1 (piecewise linear) polynomial.
    
    Parameters
    ----------
    n_cdfs:
        The number of CDFs the object contains information on.
    poly_coef:
        A tensor containing coefficients of the cubic polynomials used 
        to define each CDF in each element of the grid.
    cdf_poly_grid:
        A matrix where the number of rows is equal to the number 
        of nodes of the polynomial basis for the CDF, and the 
        number of columns is equal to the number of CDFs. Element
        (i, j) contains the value of the jth CDF at the ith node.
    poly_norm:
        A vector containing the normalising constant for each CDF.
    
    """
    n_cdfs: int
    poly_coef: Tensor
    cdf_poly_grid: Tensor
    poly_norm: Tensor


@dataclasses.dataclass
class CDFDataPiecewiseCheby(CDFData):
    """Class containing information on a single CDF, or set of CDFs, 
    for a LagrangeP or CubicHermite polynomial.
    
    Parameters
    ----------
    n_cdfs:
        The number of CDFs the object contains information on.
    poly_coef:
        A tensor containing coefficients of the Chebyshev polynomials 
        used to define each CDF in each element of the grid.
    cdf_poly_grid:
        A matrix where the number of rows is equal to the number 
        of nodes of the polynomial basis for the CDF, and the 
        number of columns is equal to the number of CDFs. Element
        (i, j) contains the value of the jth CDF at the ith node.
    poly_norm:
        A vector containing the normalising constant for each CDF.
    cdf_poly_nodes:
        A matrix containing the values of the CDF at each of the nodes 
        of the Chebyshev polynomials used to parametrise it.
    poly_base:
        A matrix containing the values of each Chebyshev polynomial at 
        left-hand edge of the element it correpsonds to.
    
    """
    n_cdfs: int
    poly_coef: Tensor
    cdf_poly_grid: Tensor
    poly_norm: Tensor
    cdf_poly_nodes: Tensor
    poly_base: Tensor