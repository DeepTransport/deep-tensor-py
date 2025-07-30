from typing import Tuple

import torch
from torch import Tensor

from .lagrange_1 import Lagrange1
from ..cdf_data import CDFDataLagrange1
from ..piecewise.piecewise_cdf import PiecewiseCDF


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

        dl = 0.5 * self.elem_size
        self.nodes = torch.linspace(*self.domain, 2 * self.num_elems + 1)

        self.node2elem = self._build_node2elem(self.cardinality, self.num_elems)
        self.V_inv = self._build_V_inv(dl)

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
        ii = range(3*n_elems)
        jj = torch.tensor([[2*i, 2*i+1, 2*i+2] for i in range(n_elems)]).flatten()
        node2elem = torch.zeros((3*n_elems, n_nodes))
        node2elem[ii, jj] = 1.0
        return node2elem
    
    @staticmethod
    def _build_V_inv(dl: float) -> Tensor:
        """Constructs the inverse of the Vandermonde matrix obtained by 
        evaluating (1, x, x^2) at (0, dl/2, dl). When applied to a 
        vector containing three equispaced function values on an 
        element, this matrix returns the coefficients of the quadratic 
        that passes through them.
        """
        V_inv = torch.tensor([
            [1.0, 0.0, 0.0], 
            [-1.5/dl, 2.0/dl, -0.5/dl], 
            [0.5/(dl**2), -1.0/(dl**2), 0.5/(dl**2)]
        ])
        return V_inv

    def pdf2cdf(self, ps: Tensor) -> CDFDataLagrange1:

        # Handle case where a vector for a single PDF is passed in
        if ps.ndim == 1:
            ps = ps[:, None]

        n_cdfs = ps.shape[1]
        
        # Compute coefficients of quadratic for each element
        ps_elems = (self.node2elem @ ps).reshape(self.num_elems, 3, n_cdfs)
        poly_coef = torch.einsum("kl, ilj", self.V_inv, ps_elems)

        temp = torch.tensor([
            self.elem_size, 
            (self.elem_size ** 2) / 2.0, 
            (self.elem_size ** 3) / 3.0
        ])

        # Integrate each quadratic polynomial over its element
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

        j_inds = torch.arange(cdf_data.n_cdfs)

        dls = (ls - self.grid[inds_left])[:, None]
        dls_mat = torch.hstack((dls, (dls**2) / 2.0, (dls**3) / 3.0))

        zs_left = cdf_data.cdf_poly_grid[inds_left, j_inds]
        coefs = cdf_data.poly_coef[inds_left, j_inds, :]
        zs = zs_left + (dls_mat * coefs).sum(dim=1)
        return zs

    def eval_int_elem_deriv(
        self, 
        cdf_data: CDFDataLagrange1, 
        inds_left: Tensor,
        ls: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        j_inds = torch.arange(cdf_data.n_cdfs)
        zs = self.eval_int_elem(cdf_data, inds_left, ls)

        dls = (ls - self.grid[inds_left])[:, None]
        dls_mat = torch.hstack((torch.ones_like(dls), dls, dls**2))
        
        coefs = cdf_data.poly_coef[inds_left, j_inds, :]
        dzdls = (dls_mat * coefs).sum(dim=1)
        return zs, dzdls