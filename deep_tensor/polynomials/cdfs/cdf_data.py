import abc
import dataclasses

import torch


@dataclasses.dataclass
class CDFData(abc.ABC):
    n_cdfs: int
    poly_coef: torch.Tensor
    cdf_poly_grid: torch.Tensor
    poly_norm: torch.Tensor
    

@dataclasses.dataclass
class CDFDataLagrange1(CDFData):
    """Class containing information on a single CDF, or set of CDFs, 
    for a Lagrange1 (piecewise linear) polynomial.
    
    Parameters
    ----------
    n_cdfs:
        The number of CDFs the class contains information on.
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
    poly_coef: torch.Tensor
    cdf_poly_grid: torch.Tensor
    poly_norm: torch.Tensor


@dataclasses.dataclass
class CDFDataLagrangeP(CDFData):
    """Class containing information on a single CDF, or set of CDFs, 
    for a LagrangeP (piecewise) polynomial.
    
    TODO: finish.
    """

    n_cdfs: int
    poly_coef: torch.Tensor
    cdf_poly_grid: torch.Tensor
    poly_norm: torch.Tensor
    cdf_poly_nodes: torch.Tensor
    poly_base: torch.Tensor