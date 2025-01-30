from typing import Tuple

import torch

from .cdf_data import CDFData
from .lagrange_1 import Lagrange1
from .piecewise_cdf import PiecewiseCDF


class Lagrange1CDF(Lagrange1, PiecewiseCDF):

    def __init__(self, poly: Lagrange1, **kwargs):
        """TODO: write.
        
        References
        ----------
        https://en.wikipedia.org/wiki/Finite_difference_coefficient
        
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

        # Define set of forward finite difference operators
        dl = self.elem_size / 2.0
        value = [1.0, 0.0, 0.0]
        first_deriv = [-1.5/dl, 2.0/dl, -0.5/dl]
        second_deriv = [0.5/(dl**2), -1.0/(dl**2), 0.5/(dl**2)]
        self._diffs = torch.tensor([value, first_deriv, second_deriv])

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
    def diffs(self) -> torch.Tensor:
        return self._diffs

    def pdf2cdf(
        self, 
        pls: torch.Tensor
    ) -> CDFData:

        # Handle case where a vector for a single PDF is passed in
        if pls.ndim == 1:
            pls = pls[:, None]

        n_cdfs = pls.shape[1]
            
        # Compute value and first two derivatives of PDF at left-hand 
        # side of each element
        # print(self.T.to_dense()
        poly_coef = self.diffs @ (self.node2elem @ pls).T.reshape(-1, 3).T 

        temp = torch.tensor([
            self.elem_size, 
            (self.elem_size ** 2) / 2.0, 
            (self.elem_size ** 3) / 3.0
        ])

        cdf_elems = (temp @ poly_coef).reshape(n_cdfs, self.num_elems).T
        cdf_poly_grid = torch.zeros(self.num_elems+1, n_cdfs)
        cdf_poly_grid[1:] = torch.cumsum(cdf_elems, dim=0)
        poly_norm = cdf_poly_grid[-1]

        return CDFData(n_cdfs, poly_coef, cdf_poly_grid, poly_norm) 

    def eval_int_lag_local(
        self, 
        cdf_data: CDFData,
        inds_left: torch.Tensor,
        ls: torch.Tensor
    ) -> torch.Tensor:
        
        if cdf_data.num_samples == 1:
            i_inds = inds_left
            j_inds = inds_left 
        else:
            coi = torch.arange(ls.numel())
            i_inds = inds_left + coi * self.num_elems 
            j_inds = inds_left + coi * (self.num_elems + 1)

        dls = (ls - self.grid[inds_left])[:, None]
        # TODO: figure out what this is
        x_mat = torch.hstack((dls, (dls**2) / 2.0, (dls**3) / 3.0))

        zs = torch.sum(x_mat * cdf_data.poly_coef[:, i_inds].T, 1)
        zs += cdf_data.cdf_poly_grid.T.flatten()[j_inds]
        return zs

    def eval_int_lag_local_deriv(
        self, 
        cdf_data: CDFData, 
        inds_left: torch.Tensor,
        ls: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        zs = self.eval_int_lag_local(cdf_data, inds_left, ls)

        dls = ls - self.grid[inds_left]

        # TODO: figure out what this is
        # (derivatives of the above)
        temp = torch.hstack((
            torch.ones((dls.shape[0], 1)),
            dls[:, None],
            dls[:, None] ** 2
        ))

        if cdf_data.num_samples == 1:
            i_inds = inds_left
        else:
            coi = torch.arange(ls.numel())
            i_inds = inds_left + coi * self.num_elems 

        dzs = torch.sum(temp * cdf_data.poly_coef[:, i_inds].T, dim=1)
        return zs, dzs