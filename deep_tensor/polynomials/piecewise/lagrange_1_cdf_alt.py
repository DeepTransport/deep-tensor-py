from typing import Tuple

import torch
from torch import Tensor

from .lagrange_1 import Lagrange1
from .piecewise_cdf import PiecewiseCDF
from ..cdf_data import CDFDataLagrange1


class Lagrange1CDF(PiecewiseCDF):

    def __init__(self, poly: Lagrange1, **kwargs):
        """The CDF for piecewise linear Lagrange polynomials.

        The CDF is a piecewise cubic polynomial (the corresponding FTT,
        which uses a piecewise linear interpolation basis, is squared, 
        then integrated). Constructing the CDF amounts to computing the 
        coefficients of each polynomial corresponding to each element 
        using three evaluations of the polynomial (at the left-hand 
        edge, centre, and right-hand edge). Inverting the CDF amounts 
        to determining which element a given CDF value lies in and then 
        inverting the corresponding polynomial (done using Newton's 
        method).

        Parameters
        ----------
        poly:
            The interpolating polynomial for the corresponding PDF.
        **kwargs:
            Arguments to pass into PiecewiseCDF.__init__.
            
        """

        PiecewiseCDF.__init__(self, **kwargs)
        
        self._poly = poly
        self.elem_size = self._poly.elem_size
        self.nodes = self._poly.nodes

        self.node2elem = self._build_node2elem(self.cardinality, self.num_elems)
        self.V_inv = self._build_V_inv(float(self.elem_size))  # TODO: fix type

        return
    
    @property
    def cardinality(self) -> int:
        return self.nodes.numel()
    
    @property 
    def domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0])
    
    @property
    def grid(self) -> Tensor:
        return self._poly.grid
    
    @property
    def num_elems(self) -> int:
        return self._poly.num_elems
    
    @property
    def node2elem(self) -> Tensor:
        """An operator which takes a vector of coefficients for the 
        nodes of the polynomial basis for the CDF, and returns a vector 
        containing the three coefficients for each element of the 
        polynomial basis for the PDF, in sequence.
        """
        return self._node2elem
    
    @node2elem.setter
    def node2elem(self, value: Tensor) -> None:
        self._node2elem = value 
        return
    
    @staticmethod 
    def _build_node2elem(n_nodes: int, n_elems: int) -> Tensor:
        ii = range(2*n_elems)
        jj = torch.tensor([[i, i+1] for i in range(n_elems)]).flatten()
        node2elem = torch.zeros((2*n_elems, n_nodes))
        node2elem[ii, jj] = 1.0
        return node2elem
    
    @staticmethod
    def _build_V_inv(dl: float) -> Tensor:
        """Constructs the inverse of the Vandermonde matrix obtained by 
        evaluating (1, l, l^2) at (0, dl/2, dl). When applied to a 
        vector containing three equispaced function values on an 
        element, this matrix returns the coefficients of the quadratic 
        that passes through them.
        """
        # V = torch.tensor([[1.0, 0.0], 
        #                   [1.0, dl]])
        # V_inv = torch.linalg.inv(V) # TODO: get the analytic version of this.
        V_inv = torch.tensor([[1.0, 0.0], [-1.0 / dl, 1.0 / dl]])
        # V_inv = torch.tensor([
        #     [1.0, 0.0, 0.0], 
        #     [-1.5/dl, 2.0/dl, -0.5/dl], 
        #     [0.5/(dl**2), -1.0/(dl**2), 0.5/(dl**2)]
        # ])
        return V_inv

    def pdf2cdf(self, ps: Tensor) -> CDFDataLagrange1:

        # Handle case where a vector for a single PDF is passed in
        if ps.ndim == 1:
            ps = ps[:, None]

        n_cdfs = ps.shape[1]
        
        # Compute coefficients of linear polynomial for each element
        ps_elems = (self.node2elem @ ps).reshape(self.num_elems, 2, n_cdfs)
        poly_coef = torch.einsum("kl, ilj", self.V_inv, ps_elems)

        temp = torch.tensor([self.elem_size, 0.5 * (self.elem_size ** 2)])

        # Integrate each linear polynomial over its element
        cdf_elems = torch.einsum("jkl, l", poly_coef, temp)

        cdf_poly_grid = torch.zeros(self.num_elems+1, n_cdfs)
        cdf_poly_grid[1:] = torch.cumsum(cdf_elems, dim=0)
        poly_norm = cdf_poly_grid[-1]

        return CDFDataLagrange1(n_cdfs, poly_coef, cdf_poly_grid, poly_norm) 

    def eval_int_elem(
        self, 
        cdf_data: CDFDataLagrange1,
        inds_left: Tensor,
        ls: Tensor
    ) -> Tensor:

        jj = torch.arange(cdf_data.n_cdfs)

        dls = (ls - self.grid[inds_left])[:, None]
        dls_mat = torch.hstack((dls, (dls**2) / 2.0))

        zs_left = cdf_data.cdf_poly_grid[inds_left, jj]
        coefs = cdf_data.poly_coef[inds_left, jj, :]
        zs = zs_left + (dls_mat * coefs).sum(dim=1)
        return zs

    def eval_int_elem_deriv(
        self, 
        cdf_data: CDFDataLagrange1, 
        inds_left: Tensor,
        ls: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        jj = torch.arange(cdf_data.n_cdfs)
        zs = self.eval_int_elem(cdf_data, inds_left, ls)

        dls = (ls - self.grid[inds_left])[:, None]
        dls_mat = torch.hstack((torch.ones_like(dls), dls))
        
        coefs = cdf_data.poly_coef[inds_left, jj, :]
        dzdls = (dls_mat * coefs).sum(dim=1)
        return zs, dzdls
    
    def invert_cdf_elem(
        self, 
        cdf_data: CDFDataLagrange1, 
        inds_left: Tensor,
        zs_cdf: Tensor
    ) -> Tensor:
        """Evaluates the inverse of the CDF corresponding to the 
        (unnormalised) target PDF at a given set of values.
        
        Parameters
        ----------
        cdf_data:
            An object containing information about the properties of 
            the CDF.
        inds_left:
            An n-dimensional vector containing the indices of the 
            points of the grid on which the target PDF is discretised 
            that are immediately to the left of each value in ls.
        zs_cdf:
            An n-dimensional vector of values in the range [0, Z], 
            where Z is the normalising constant associated with the 
            (unnormalised) target PDF.

        Returns
        -------
        ls:
            An n-dimensional vector containing the inverse of the CDF 
            evaluated at each element in zs_cdf.
        
        """

        cdf_data.poly_coef

        l0s, l1s = self.grid[inds_left], self.grid[inds_left+1]
        ls = self.newton(cdf_data, inds_left, zs_cdf, l0s, l1s)
        return ls