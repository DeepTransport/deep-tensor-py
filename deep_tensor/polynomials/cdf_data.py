import torch


class CDFData():

    def __init__(
        self, 
        n_cdfs: int,
        poly_coef: torch.Tensor, 
        cdf_poly_grid: torch.Tensor, 
        poly_norm: torch.Tensor
    ):
        """Class containing information on a single CDF or set of CDFs.

        TODO: rename to CDFDataLagrange1
        
        Parameters
        ----------
        n_cdfs:
            The number of CDFs the class contains information on.
        poly_coef:
            TODO: figure out what is going on here. In 
        cdf_poly_grid:
            A matrix where the number of rows is equal to the number 
            of nodes of the polynomial basis for the CDF, and the 
            number of columns is equal to the number of CDFs. Element
            (i, j) contains the value of the jth CDF at the ith node.
        poly_norm:
            A vector containing the normalising constant for each CDF.
        
        """
        
        self.n_cdfs = n_cdfs
        self.poly_coef = poly_coef
        self.cdf_poly_grid = cdf_poly_grid
        self.poly_norm = poly_norm
        return
    
class CDFDataLagrangeP():

    def __init__(
        self,
        n_cdfs: int,
        poly_coef: torch.Tensor,
        cdf_poly_nodes: torch.Tensor,
        cdf_poly_grid: torch.Tensor,
        poly_norm: torch.Tensor,
        poly_base: torch.Tensor
    ):
        """TODO: write docstring."""
        
        self.n_cdfs = n_cdfs 
        self.poly_coef = poly_coef
        self.cdf_poly_nodes = cdf_poly_nodes 
        self.cdf_poly_grid = cdf_poly_grid
        self.poly_norm = poly_norm
        self.poly_base = poly_base
        return