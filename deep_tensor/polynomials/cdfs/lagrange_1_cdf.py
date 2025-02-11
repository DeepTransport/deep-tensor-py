from typing import Tuple

import torch

from .cdf_data import CDFDataLagrange1
from .piecewise_cdf import PiecewiseCDF
from ..polynomials.lagrange_1 import Lagrange1


class Lagrange1CDF(Lagrange1, PiecewiseCDF):

    def __init__(self, poly: Lagrange1, **kwargs):
        """The CDF for piecewise linear Lagrange polynomials.

        Parameters
        ----------
        poly:
            The interpolating polynomial for the corresponding PDF.
        **kwargs:
            Arguments to pass into PiecewiseCDF.__init__.
            
        """
        
        Lagrange1.__init__(self, num_elems=poly.num_elems)
        PiecewiseCDF.__init__(self, **kwargs)
        
        num_nodes = 2 * self.num_elems + 1
        self._nodes = torch.linspace(*self.domain, num_nodes)

        ii = torch.tensor([[3*i, 3*i+1, 3*i+2] for i in range(self.num_elems)])
        jj = torch.tensor([[2*i, 2*i+1, 2*i+2] for i in range(self.num_elems)])
        indices = torch.vstack((ii.flatten(), jj.flatten()))
        values = torch.ones(3 * self.num_elems)
        shape = (3 * self.num_elems, num_nodes)
        self._node2elem = torch.sparse_coo_tensor(indices, values, shape)

        dl = self.elem_size / 2.0
        self._V_inv = torch.tensor([
            [1.0, 0.0, 0.0], 
            [-1.5/dl, 2.0/dl, -0.5/dl], 
            [0.5/(dl**2), -1.0/(dl**2), 0.5/(dl**2)]
        ])

        return

    @property 
    def nodes(self) -> torch.Tensor:
        return self._nodes
    
    @property
    def node2elem(self) -> torch.Tensor:
        """An operator which takes a vector of coefficients for the 
        nodes of the polynomial basis for the CDF, and returns a vector 
        containing the three coefficients for each element of the 
        polynomial basis for the PDF, in sequence.
        """
        return self._node2elem

    @property
    def V_inv(self) -> torch.Tensor:
        """The inverse of the Vandermonde matrix obtained by evaluating 
        (1, x, x^2) at (0, dl/2, dl).
        """
        return self._V_inv

    def pdf2cdf(
        self, 
        pls: torch.Tensor
    ) -> CDFDataLagrange1:

        # Handle case where a vector for a single PDF is passed in
        if pls.ndim == 1:
            pls = pls[:, None]

        n_cdfs = pls.shape[1]
        
        # Compute coefficients of (quadratic) polynomial used to define
        # PDF in each element
        poly_coef = self.V_inv @ (self.node2elem @ pls).T.reshape(-1, 3).T 

        temp = torch.tensor([
            self.elem_size, 
            (self.elem_size ** 2) / 2.0, 
            (self.elem_size ** 3) / 3.0
        ])

        # Compute the integral of each quadratic polynomial over its element
        cdf_elems = (temp @ poly_coef).reshape(n_cdfs, self.num_elems).T

        cdf_poly_grid = torch.zeros(self.num_elems+1, n_cdfs)
        cdf_poly_grid[1:] = torch.cumsum(cdf_elems, dim=0)
        poly_norm = cdf_poly_grid[-1]

        return CDFDataLagrange1(n_cdfs, poly_coef, cdf_poly_grid, poly_norm) 

    def eval_int_lag_local(
        self, 
        cdf_data: CDFDataLagrange1,
        inds_left: torch.Tensor,
        ls: torch.Tensor
    ) -> torch.Tensor:
        
        if cdf_data.n_cdfs == 1:
            i_inds = inds_left
            j_inds = inds_left 
        else:
            coi = torch.arange(ls.numel())
            i_inds = inds_left + coi * self.num_elems 
            j_inds = inds_left + coi * (self.num_elems + 1)

        dls = (ls - self.grid[inds_left])[:, None]
        dls_mat = torch.hstack((dls, (dls**2) / 2.0, (dls**3) / 3.0))

        zs = torch.sum(dls_mat * cdf_data.poly_coef[:, i_inds].T, 1)
        zs += cdf_data.cdf_poly_grid.T.flatten()[j_inds]
        return zs

    def eval_int_lag_local_deriv(
        self, 
        cdf_data: CDFDataLagrange1, 
        inds_left: torch.Tensor,
        ls: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        zs = self.eval_int_lag_local(cdf_data, inds_left, ls)

        dls = (ls - self.grid[inds_left])[:, None]
        dls_mat = torch.hstack((torch.ones_like(dls), dls, dls ** 2))

        if cdf_data.n_cdfs == 1:
            i_inds = inds_left
        else:
            coi = torch.arange(ls.numel())
            i_inds = inds_left + coi * self.num_elems 

        dzs = torch.sum(dls_mat * cdf_data.poly_coef[:, i_inds].T, dim=1)
        return zs, dzs