import abc
from typing import Tuple

import torch
from torch import Tensor

from .piecewise import Piecewise
from ..cdf_data import CDFDataPiecewiseCheby
from ..piecewise.piecewise_cdf import PiecewiseCDF
from ..spectral.chebyshev_2nd_cdf import Chebyshev2ndCDF
from ..spectral.chebyshev_2nd import Chebyshev2nd


class PiecewiseChebyshevCDF(PiecewiseCDF, abc.ABC):

    def __init__(self, poly: Piecewise, error_tol: float):
        
        PiecewiseCDF.__init__(self, error_tol, poly._device)

        self._mass = None 
        self._mass_R = None 
        self._int_W = None

        # Define local CDF polynomial
        self._piecewise2cheby(poly)
        self._compute_cdf_nodes()

        n_cheby = self._cheby._cardinality
        self._elem_nodes = torch.tensor([
            range(n*n_cheby-n, (n+1)*n_cheby-n) 
            for n in range(self._num_elems)], device=self._device)
        
        return

    def _piecewise2cheby(self, poly) -> None:
        """Defines a data structure which maps higher-order piecewise 
        polynomials (i.e., LagrangeP or CubicHermite) to Chebyshev 
        polynomials with preserved boundary values. 

        Parameters
        ----------
        poly:
            A higher-order piecewise polynomial basis.
        
        """

        self._cheby = Chebyshev2ndCDF(poly, error_tol=self._error_tol)
        assert self._cheby._cardinality > 3, "Must use more than three nodes."

        cheby_nodes = Chebyshev2nd(self._cheby._cardinality-3)._nodes
        ref_nodes = [self._cheby._domain[0], *cheby_nodes, self._cheby._domain[1]]
        ref_nodes = torch.tensor(ref_nodes, device=self._device)

        self._cheby._basis2node = self._cheby._eval_basis(ref_nodes)
        self._cheby._node2basis = torch.linalg.inv(self._cheby._basis2node)
        self._cheby._nodes = 0.5 * (ref_nodes + 1.0)  # map nodes into [0, 1]
        self._cheby._mass_R = None 
        self._cheby._int_W = None

        self._cdf_basis2node = 0.5 * self._cheby._eval_int_basis(ref_nodes)
        return

    def _compute_cdf_nodes(self) -> None:
        """Computes the collocation points in each element."""
        n_cheby = self._cheby._cardinality
        n_nodes = self._num_elems * (n_cheby - 1) + 1
        nodes = torch.zeros(n_nodes, device=self._device)
        for i in range(self._num_elems):
            inds = torch.arange(n_cheby, device=self._device) + i * (n_cheby - 1)
            nodes[inds] = self._grid[i] + self._cheby._nodes * self._elem_size
        self._nodes = nodes
        return
    
    def _pdf2cdf(self, ps: Tensor) -> CDFDataPiecewiseCheby:

        n_cdfs = ps.shape[1]

        # Form tensor containing the value of the PDF at each node in 
        # each element
        shape = (self._num_elems, self._cheby._cardinality, n_cdfs)
        ps_local = ps[self._elem_nodes.flatten(), :].reshape(*shape)
        
        # Compute the coefficients of each Chebyshev polynomial in each 
        # element for each PDF
        poly_coef = torch.einsum("jl, ilk -> ijk", self._cheby._node2basis, ps_local)

        cdf_poly_grid = torch.zeros(self._num_elems+1, n_cdfs, device=self._device)
        cdf_poly_nodes = torch.zeros(self._cardinality, n_cdfs, device=self._device)
        poly_base = torch.zeros(self._num_elems, n_cdfs, device=self._device)

        for i in range(self._num_elems):

            # Compute values of integral of Chebyshev polynomial over 
            # current element
            integrals = (self._cdf_basis2node @ poly_coef[i, :, :]) * self._jac
            # Compute value of CDF poly at LHS of element
            poly_base[i] = integrals[0]
            # Compute value of poly at nodes of CDF corresponding to 
            # current element
            cdf_poly_nodes[self._elem_nodes[i]] = cdf_poly_grid[i] + integrals - integrals[0]
            # Compute value of CDFs at the right-hand edge of element
            cdf_poly_grid[i+1] = cdf_poly_grid[i] + integrals[-1] - integrals[0]
        
        # Compute normalising constant
        poly_norm = cdf_poly_grid[-1]

        data = CDFDataPiecewiseCheby(
            n_cdfs, 
            poly_coef, 
            cdf_poly_grid, 
            poly_norm, 
            cdf_poly_nodes, 
            poly_base
        )

        return data

    def _eval_int_elem(
        self, 
        cdf_data: CDFDataPiecewiseCheby, 
        inds_left: Tensor, 
        ls: Tensor 
    ) -> Tensor:

        # Rescale each element of ls to interval [-1, 1]
        mid = 0.5 * (self._grid[inds_left] + self._grid[inds_left+1])
        ls = (ls - mid) / (0.5 * self._jac)

        j_inds = torch.arange(cdf_data.n_cdfs, device=self._device)
        ps = self._cheby._eval_int_basis(ls) * 0.5 * self._jac

        coefs = cdf_data.poly_coef[inds_left, :, j_inds]
        zs_left = (cdf_data.cdf_poly_grid[inds_left, j_inds] 
                   - cdf_data.poly_base[inds_left, j_inds])

        zs = zs_left + (ps * coefs).sum(dim=1)
        return zs
    
    def _eval_int_elem_deriv(
        self, 
        cdf_data: CDFDataPiecewiseCheby, 
        inds_left: Tensor, 
        ls: Tensor
    ) -> Tuple[Tensor, Tensor]:

        # Rescale each element of ls to interval [-1, 1]
        mid = 0.5 * (self._grid[inds_left] + self._grid[inds_left+1])
        ls = (ls - mid) / (0.5 * self._jac)
        ls = torch.clamp(ls, -1.0, 1.0)

        j_inds = torch.arange(cdf_data.n_cdfs, device=self._device)
        ps, dpdls = self._cheby._eval_int_basis_newton(ls)
        ps *= (0.5 * self._jac)

        coefs = cdf_data.poly_coef[inds_left, :, j_inds]
        zs_left = (cdf_data.cdf_poly_grid[inds_left, j_inds] 
                   - cdf_data.poly_base[inds_left, j_inds])

        zs = zs_left + (ps * coefs).sum(dim=1)
        dzdls = (dpdls * coefs).sum(dim=1)
        return zs, dzdls
    
    def _invert_cdf_elem(
        self, 
        cdf_data: CDFDataPiecewiseCheby, 
        inds_left: Tensor, 
        zs_cdf: Tensor
    ) -> Tensor:

        inds = (cdf_data.cdf_poly_nodes < zs_cdf).sum(dim=0) - 1
        inds = torch.clamp(inds, 0, self._cardinality-2)
        
        l0s = self._nodes[inds]
        l1s = self._nodes[inds+1]
        ls = self._newton(cdf_data, inds_left, zs_cdf, l0s, l1s)
        return ls