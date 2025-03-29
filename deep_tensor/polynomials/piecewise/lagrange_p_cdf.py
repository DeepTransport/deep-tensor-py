from typing import Tuple

import torch
from torch import Tensor

from .lagrange_p import LagrangeP
from ..cdf_data import CDFDataLagrangeP
from ..piecewise.piecewise_cdf import PiecewiseCDF
from ..spectral.chebyshev_2nd_cdf import Chebyshev2ndCDF
from ..spectral.chebyshev_2nd import Chebyshev2nd


class LagrangePCDF(LagrangeP, PiecewiseCDF):

    def __init__(self, poly: LagrangeP, **kwargs):

        LagrangeP.__init__(self, poly.order, poly.num_elems)
        PiecewiseCDF.__init__(self, **kwargs)

        # Define local CDF polynomial
        self.cheby, self.cdf_basis2node = self._lag2cheby(poly)

        self.mass = None 
        self.mass_R = None 
        self.int_W = None
        self._compute_cdf_nodes()

        n_cheby = self.cheby.cardinality
        self.elem_nodes = torch.tensor([
            range(n*n_cheby-n, (n+1)*n_cheby-n) 
            for n in range(self.num_elems)])

        return
    
    def _compute_cdf_nodes(self) -> None:
        """Computes the collocation points in each element."""
        n_cheby = self.cheby.cardinality
        n_nodes = self.num_elems * (n_cheby - 1) + 1
        nodes = torch.zeros(n_nodes)
        for i in range(self.num_elems):
            inds = torch.arange(n_cheby) + i * (n_cheby - 1)
            nodes[inds] = self.grid[i] + self.cheby.nodes * self.elem_size
        self.nodes = nodes
        return

    def _lag2cheby(self, poly: LagrangeP) -> Tuple[Chebyshev2ndCDF, Tensor]:
        """Defines a data structure which maps Lagrange polynomials to 
        Chebyshev polynomials with preserved boundary values. 

        Parameters
        ----------
        poly:
            A LagrangeP polynomial basis.
        
        Returns
        -------
        TODO
        
        """

        cheby = Chebyshev2ndCDF(poly)
        n_nodes = cheby.cardinality

        assert n_nodes > 3, "Must use more than three nodes."

        cheby_2nd = Chebyshev2nd(n_nodes-3)
        ref_nodes = [cheby.domain[0], *cheby_2nd.nodes, cheby.domain[1]]
        ref_nodes = torch.tensor(ref_nodes)

        cheby.basis2node = cheby.eval_basis(ref_nodes)
        cheby.node2basis = torch.linalg.inv(cheby.basis2node)
        cheby.nodes = 0.5 * (ref_nodes + 1.0)  # map nodes into [0, 1]
        cheby.mass_R = None 
        cheby.int_W = None

        cdf_basis2node = 0.5 * cheby.eval_int_basis(ref_nodes)
        return cheby, cdf_basis2node
    
    def pdf2cdf(self, ps: Tensor) -> CDFDataLagrangeP:

        # Handle case where a single PDF is passed in
        if ps.ndim == 1:
            ps = ps[:, None]

        n_cdfs = ps.shape[1]

        # Form tensor containing the value of the PDF at each node in 
        # each element
        shape = (self.num_elems, self.cheby.cardinality, n_cdfs)
        ps_local = ps[self.elem_nodes.flatten(), :].reshape(*shape)
        
        # Compute the coefficients of each Chebyshev polynomial in each 
        # element for each PDF
        poly_coef = torch.einsum("jl, ilk -> ijk", self.cheby.node2basis, ps_local)

        cdf_poly_grid = torch.zeros(self.num_elems+1, n_cdfs)
        cdf_poly_nodes = torch.zeros(self.cardinality, n_cdfs)
        poly_base = torch.zeros(self.num_elems, n_cdfs)

        for i in range(self.num_elems):

            # Compute values of integral of Chebyshev polynomial over 
            # current element
            integrals = (self.cdf_basis2node @ poly_coef[i, :, :]) * self.jac
            # Compute value of CDF poly at LHS of element
            poly_base[i] = integrals[0]
            # Compute value of poly at nodes of CDF corresponding to 
            # current element
            cdf_poly_nodes[self.elem_nodes[i]] = cdf_poly_grid[i] + integrals - integrals[0]
            # Compute value of CDFs at the right-hand edge of element
            cdf_poly_grid[i+1] = cdf_poly_grid[i] + integrals[-1] - integrals[0]
        
        # Compute normalising constant
        poly_norm = cdf_poly_grid[-1]

        data = CDFDataLagrangeP(
            n_cdfs, 
            poly_coef, 
            cdf_poly_grid, 
            poly_norm, 
            cdf_poly_nodes, 
            poly_base
        )

        return data

    def eval_int_local(
        self, 
        cdf_data: CDFDataLagrangeP, 
        inds_left: Tensor, 
        ls: Tensor 
    ) -> Tensor:

        # Rescale each element of ls to interval [-1, 1]
        mid = 0.5 * (self.grid[inds_left] + self.grid[inds_left+1])
        ls = (ls - mid) / (0.5 * self.jac)

        j_inds = torch.arange(cdf_data.n_cdfs)
        ps = self.cheby.eval_int_basis(ls) * 0.5 * self.jac

        coefs = cdf_data.poly_coef[inds_left, :, j_inds]
        zs_left = (cdf_data.cdf_poly_grid[inds_left, j_inds] 
                   - cdf_data.poly_base[inds_left, j_inds])

        zs = zs_left + (ps * coefs).sum(dim=1)
        return zs
    
    def eval_int_local_deriv(
        self, 
        cdf_data: CDFDataLagrangeP, 
        inds_left: Tensor, 
        ls: Tensor
    ) -> Tuple[Tensor, Tensor]:

        # Rescale each element of ls to interval [-1, 1]
        mid = 0.5 * (self.grid[inds_left] + self.grid[inds_left+1])
        ls = (ls - mid) / (0.5 * self.jac)

        j_inds = torch.arange(cdf_data.n_cdfs)
        ps, dpdls = self.cheby.eval_int_basis_newton(ls)
        ps *= (0.5 * self.jac)

        coefs = cdf_data.poly_coef[inds_left, :, j_inds]
        zs_left = (cdf_data.cdf_poly_grid[inds_left, j_inds] 
                   - cdf_data.poly_base[inds_left, j_inds])

        zs = zs_left + (ps * coefs).sum(dim=1)
        dzdls = (dpdls * coefs).sum(dim=1)
        return zs, dzdls
    
    def invert_cdf_local(
        self, 
        cdf_data: CDFDataLagrangeP, 
        inds_left: Tensor, 
        zs_cdf: Tensor
    ) -> Tensor:

        inds = (cdf_data.cdf_poly_nodes < zs_cdf).sum(dim=0) - 1
        inds = torch.clamp(inds, 0, self.cardinality-2)
        
        l0s = self.nodes[inds]
        l1s = self.nodes[inds+1]
        ls = self.newton(cdf_data, inds_left, zs_cdf, l0s, l1s)
        return ls