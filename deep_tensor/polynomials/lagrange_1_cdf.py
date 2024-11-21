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

        self._iV = torch.tensor([
            [1.0, 0.0, 0.0],
            [-3.0 / (2.0*dhh), 2.0 / dhh, -1.0 / (2*dhh)], 
            [1.0 / (2.0*dhh**2), -1.0 / (dhh**2), 1.0 / (2.0*dhh**2)]
        ])
        
        indices = torch.vstack((ii.flatten(), jj.flatten()))
        values = torch.ones(3 * self.num_elems)
        shape = (3 * self.num_elems, num_nodes)

        # TODO: figure out what this is for.
        self._T = torch.sparse_coo_tensor(indices, values, shape)
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
        pdf_vals: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""

        # TEMP
        # pdf_vals = reshape_matlab(torch.arange(pdf_vals.numel(), dtype=torch.float32)+1, pdf_vals.shape)

        pdf_vals = torch.atleast_2d(pdf_vals)
        num_samples = pdf_vals.shape[1]
            
        poly_coef = self.iV @ reshape_matlab(self.T @ pdf_vals, (3, -1)) 

        temp = torch.tensor([self.elem_size, (self.elem_size**2)/2.0, (self.elem_size**3)/3.0])
        cdf_elems = temp @ reshape_matlab(poly_coef, (3, -1))
        cdf_elems = reshape_matlab(cdf_elems, (self.num_elems, -1))

        cdf_poly_grid = torch.zeros(self.num_elems + 1, num_samples)
        cdf_poly_grid[1:, :] = torch.cumsum(cdf_elems, dim=0)
        
        poly_norm = cdf_poly_grid[-1, :]

        return CDFData(num_samples, poly_coef, cdf_poly_grid, poly_norm) 

    def invert_cdf_local(
        self, 
        data: CDFData, 
        e_ks: torch.Tensor,
        mask: torch.Tensor, 
        rhs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""

        x0s, x1s = self.grid[e_ks], self.grid[e_ks+1]
        rs = self.newton(data, e_ks, mask, rhs, x0s, x1s)
        return rs

    def eval_int_lag_local(
        self, 
        data: CDFData,
        e_ks: torch.Tensor,
        mask: torch.Tensor, 
        xs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write this."""

        xs = xs - reshape_matlab(self.grid[e_ks], xs.shape)
        if data.num_samples > 1:
            
            coi = mask.nonzero().flatten()
            
            i_inds = e_ks + coi * self.num_elems 
            j_inds = e_ks + coi * (self.num_elems + 1)

            # TODO: figure out that this is
            x_mat = torch.hstack((
                xs[:, None],
                (xs[:, None] ** 2) / 2.0,
                (xs[:, None] ** 3) / 3.0
            ))

            fs = torch.sum(x_mat * data.poly_coef[:, i_inds].T, 1)
            fs = fs + data.cdf_poly_grid.T.flatten()[j_inds]
        
        else:
            raise NotImplementedError()
        
        return fs

    def eval_int_lag_local_deriv(
        self, 
        data: CDFData, 
        e_ks: torch.Tensor,
        mask: torch.Tensor,
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        xs = xs - reshape_matlab(self.grid[e_ks], xs.shape)

        if data.num_samples > 1:
            
            coi = mask.nonzero().flatten()
            
            i_inds = e_ks + coi * self.num_elems 
            j_inds = e_ks + coi * (self.num_elems + 1)

            # TODO: figure out what this is
            x_mat = torch.hstack((
                xs[:, None],
                (xs[:, None] ** 2) / 2.0,
                (xs[:, None] ** 3) / 3.0
            ))
            fs = torch.sum(x_mat * data.poly_coef[:, i_inds].T, 1)
            fs = fs + data.cdf_poly_grid.T.flatten()[j_inds]

            temp = torch.hstack((
                torch.ones((xs.shape[0], 1)),
                xs[:, None],
                xs[:, None] ** 2
            ))
            dfs = torch.sum(temp * data.poly_coef[:, i_inds].T, dim=1)
        
        else:
            raise NotImplementedError()
        
        return fs, dfs

    def eval_int_deriv(self):
        raise NotImplementedError()