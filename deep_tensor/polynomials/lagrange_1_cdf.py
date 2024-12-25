from typing import Tuple

import torch

from .cdf_data import CDFData
from .lagrange_1 import Lagrange1
from .piecewise_cdf import PiecewiseCDF
from ..tools import reshape_matlab


class Lagrange1CDF(Lagrange1, PiecewiseCDF):

    def __init__(self, poly: Lagrange1, **kwargs):
        
        Lagrange1.__init__(self, num_elems=poly.num_elems)
        PiecewiseCDF.__init__(self, **kwargs)
        
        num_nodes = 2 * self.num_elems + 1
        self._nodes = torch.linspace(*self.domain, num_nodes)
        
        dhh = self.elem_size / 2.0

        ii = torch.zeros((self.num_elems, 3), dtype=torch.int32)
        jj = torch.zeros((self.num_elems, 3), dtype=torch.int32)
        for i in range(self.num_elems):
            ii[i] = 3*i + torch.arange(3)
            jj[i] = 2*i + torch.arange(3)
        
        indices = torch.vstack((ii.flatten(), jj.flatten()))
        values = torch.ones(3 * self.num_elems)
        shape = (3 * self.num_elems, num_nodes)
        self._T = torch.sparse_coo_tensor(indices, values, shape)

        # Define set of forward finite difference operators
        self._iV = torch.tensor([
            [1.0, 0.0, 0.0],
            [-3.0 / (2.0*dhh), 2.0 / dhh, -1.0 / (2*dhh)], 
            [1.0 / (2.0*dhh**2), -1.0 / (dhh**2), 1.0 / (2.0*dhh**2)]
        ])

        return

    @property 
    def nodes(self) -> torch.Tensor:
        return self._nodes
    
    @property
    def T(self) -> torch.Tensor:
        return self._T

    @property
    def iV(self) -> torch.Tensor:
        return self._iV

    def pdf2cdf(
        self, 
        pi_rs: torch.Tensor
    ) -> CDFData:
        """
        NOTE: 
        poly_coef: contains value/deriv/second deriv of PDF at each element.
        poly_norm: normalising constant.

        """

        pi_rs = torch.atleast_2d(pi_rs)
        num_samples = pi_rs.shape[1]
            
        # Value of PDF at each element, derivative, second derivative
        poly_coef = self.iV @ (self.T @ pi_rs).T.reshape(-1, 3).T 

        temp = torch.tensor([
            self.elem_size, 
            (self.elem_size ** 2) / 2.0, 
            (self.elem_size ** 3) / 3.0
        ])
        
        cdf_elems = (temp @ poly_coef).reshape(num_samples, self.num_elems).T
        cdf_poly_grid = torch.zeros(self.num_elems+1, num_samples)
        cdf_poly_grid[1:] = torch.cumsum(cdf_elems, dim=0)
        poly_norm = cdf_poly_grid[-1]

        return CDFData(num_samples, poly_coef, cdf_poly_grid, poly_norm) 

    def invert_cdf_local(
        self, 
        data: CDFData, 
        inds_left: torch.Tensor,
        mask: torch.Tensor, 
        rhs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring (in the parent class)."""

        r0s, r1s = self.grid[inds_left], self.grid[inds_left+1]
        rs = self.newton(data, inds_left, mask, rhs, r0s, r1s)
        return rs

    def eval_int_lag_local(
        self, 
        data: CDFData,
        inds_left: torch.Tensor,
        mask: torch.Tensor, 
        rs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write this."""

        drs = rs - self.grid[inds_left]
        
        if data.num_samples == 1:
            raise NotImplementedError("Does this make a difference?")
            
        coi = mask.nonzero().flatten()
        i_inds = inds_left + coi * self.num_elems 
        j_inds = inds_left + coi * (self.num_elems + 1)

        # TODO: figure out that this is
        x_mat = torch.hstack((
            drs[:, None],
            (drs[:, None] ** 2) / 2.0,
            (drs[:, None] ** 3) / 3.0
        ))

        fs = torch.sum(x_mat * data.poly_coef[:, i_inds].T, 1)
        fs += data.cdf_poly_grid.T.flatten()[j_inds]
        return fs

    def eval_int_lag_local_deriv(
        self, 
        data: CDFData, 
        e_ks: torch.Tensor,
        mask: torch.Tensor,
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        xs = xs - reshape_matlab(self.grid[e_ks], xs.shape)

        if data.num_samples == 1:
            raise NotImplementedError("Does this make a difference?")
            
        coi = mask.nonzero().flatten()
        
        i_inds = e_ks + coi * self.num_elems 
        j_inds = e_ks + coi * (self.num_elems + 1)

        # TODO: figure out what this is
        x_mat = torch.hstack((
            xs[:, None],
            (xs[:, None] ** 2) / 2.0,
            (xs[:, None] ** 3) / 3.0
        ))
        fs = torch.sum(x_mat * data.poly_coef[:, i_inds].T, dim=1)
        fs = fs + data.cdf_poly_grid.T.flatten()[j_inds]

        temp = torch.hstack((
            torch.ones((xs.shape[0], 1)),
            xs[:, None],
            xs[:, None] ** 2
        )) # (derivatives of the above)
        dfs = torch.sum(temp * data.poly_coef[:, i_inds].T, dim=1)
        
        return fs, dfs

    def eval_int_deriv(self):
        raise NotImplementedError()