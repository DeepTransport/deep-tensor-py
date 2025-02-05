from typing import Tuple

import torch

from .chebyshev_2nd_unweighted import Chebyshev2ndUnweighted
from .lagrange_p import LagrangeP

from .bounded_poly_cdf import BoundedPolyCDF
from .cdf_data import CDFDataLagrangeP
from .piecewise_cdf import PiecewiseCDF


class LagrangePCDF(LagrangeP, PiecewiseCDF):

    def __init__(self, poly: LagrangeP, **kwargs):

        LagrangeP.__init__(self, poly.order, poly.num_elems)
        PiecewiseCDF.__init__(self, **kwargs)

        # Define local CDF polynomial
        self.cheby, self.cdf_basis2node = self.lag2cheby(poly)

        self.mass = None 
        self.mass_R = None 
        self.int_W = None

        n_cheby = self.cheby.cardinality
        n_nodes = self.num_elems * (n_cheby - 1) + 1
        self.nodes = torch.zeros(n_nodes)

        for i in range(self.num_elems):
            inds = torch.arange(self.cheby.cardinality) + i * (n_cheby - 1)
            self.nodes[inds] = self.cheby.nodes * self.elem_size + self.grid[i]

        if n_cheby > 2:
            # Operator which maps a function evaluated at each node to 
            # the value of the function evaluated at each node of each 
            # element
            global2local = torch.arange(n_nodes-1).reshape(self.num_elems, n_cheby-1)
            self.global2local = torch.hstack((global2local, global2local[:, -1:]+1)).flatten()

        else: 
            raise NotImplementedError()

        return
    
    def lag2cheby(
        self,
        poly: LagrangeP
    ) -> Tuple[BoundedPolyCDF, torch.Tensor]:
        """Defines a data structure which maps Lagrange polynomials to 
        Chebyshev polynomials with preserved boundary values. 

        TODO: finish.
        """

        cheby = BoundedPolyCDF(poly)
        n_nodes = cheby.cardinality

        if n_nodes < 3:
            msg = "Must use more than three nodes."
            raise Exception(msg)

        cheby_2nd = Chebyshev2ndUnweighted(n_nodes-3)
        ref_nodes = [cheby.domain[0], *cheby_2nd.nodes, cheby.domain[1]]
        ref_nodes = torch.tensor(ref_nodes)

        # Map reference nodes into [0, 1]
        cheby.nodes = 0.5 * (ref_nodes + 1)
        cheby.basis2node = cheby.eval_basis(ref_nodes)
        cheby.node2basis = torch.linalg.inv(cheby.basis2node)  # TODO: tidy this up

        cheby.mass_R = None 
        cheby.int_W = None
        cdf_basis2node = 0.5 * cheby.eval_int_basis(ref_nodes)

        return cheby, cdf_basis2node
    
    def pdf2cdf(self, ps: torch.Tensor) -> CDFDataLagrangeP:
        
        n_cdfs = ps.shape[1]

        if n_cdfs == 1:
            raise NotImplementedError()

        # Form tensor containing the value of the PDF at each node in 
        # each element
        ps_local = (ps[self.global2local, :]
                    .reshape(self.num_elems, self.cheby.cardinality, n_cdfs)
                    .permute(1, 2, 0))  # TODO: get rid of permutation?
        
        # Compute the coefficients of each Chebyshev polynomial (i) in 
        # each element (k) for each PDF (j)
        poly_coef = torch.einsum("il, ljk -> ijk", self.cheby.node2basis, ps_local)
        
        cdf_poly_grid = torch.zeros(self.num_elems+1, n_cdfs)
        cdf_poly_nodes = torch.zeros(self.cardinality, n_cdfs)
        poly_base = torch.zeros(self.num_elems, n_cdfs)

        inds = self.global2local.reshape(self.num_elems, -1)

        for i in range(self.num_elems):

            # Compute nodal values of each CDF polynomial for current 
            # element
            tmp = (self.cdf_basis2node @ poly_coef[:, :, i]) * self.jac
            
            # print(self.cheby.eval_int_basis(tmp[0] / self.jac))

            assert torch.min(tmp-tmp[0]) >= -1e-10, "Mistake"  # TODO: move elsewhere

            # Compute value of CDF poly at LHS of element
            poly_base[i] = tmp[0]
            # Compute value of poly at nodes of CDF corresponding to 
            # current element
            cdf_poly_nodes[inds[i]] = cdf_poly_grid[i] + tmp - tmp[0]
            # Compute value of CDFs at the right-hand edge of element
            cdf_poly_grid[i+1] = cdf_poly_grid[i] + tmp[-1] - tmp[0]
        
        # Compute normalising constant
        poly_norm = cdf_poly_grid[-1]

        # from matplotlib import pyplot as plt
        # # plt.plot(ps[:, 0])
        # plt.plot(cdf_poly_nodes[:, 0])
        # plt.show()

        return CDFDataLagrangeP(n_cdfs, poly_coef, cdf_poly_nodes, cdf_poly_grid, poly_norm, poly_base)

    def eval_int_lag_local(
        self, 
        cdf_data: CDFDataLagrangeP, 
        inds_left: torch.Tensor, 
        ls: torch.Tensor 
    ) -> torch.Tensor:

        # Define the domain on which to evaluate the section of the CDF
        domains = torch.hstack((
            self.grid[inds_left][:, None],
            self.grid[inds_left+1][:, None]
        ))

        # Rescale each element of ls to interval [-1, 1]
        x2z = 0.5 * (domains[:, 1] - domains[:, 0])
        mid = 0.5 * (domains[:, 1] + domains[:, 0])
        ls = (ls - mid) / x2z

        if cdf_data.n_cdfs == 1:
            raise NotImplementedError()
        
        else:

            pi = torch.arange(cdf_data.n_cdfs)
            basis_vals = self.cheby.eval_int_basis(ls)  # TODO: check this
            basis_vals *= x2z[:, None]  # Plays similar role to Jacobian?
            tmp = (basis_vals * cdf_data.poly_coef[:, pi, inds_left].T).sum(dim=1)
            F = tmp - cdf_data.poly_base[inds_left, pi] + cdf_data.cdf_poly_grid[inds_left+1, pi]

            print(tmp - cdf_data.poly_base[inds_left, pi])
            print((tmp - cdf_data.poly_base[inds_left, pi]).min())

        return F
    
    def eval_int_lag_local_deriv(self, cdf_data, inds_left, ls):
        raise NotImplementedError()
    
    def invert_cdf_local(
        self, 
        cdf_data: CDFDataLagrangeP, 
        inds_left: torch.Tensor, 
        zs_cdf: torch.Tensor
    ) -> torch.Tensor:

        if cdf_data.n_cdfs == 1:
            raise NotImplementedError()
        else:
            inds = (cdf_data.cdf_poly_nodes < zs_cdf).sum(dim=0) - 1
        
        inds = torch.clamp(inds, 0, self.cardinality-2)
        l0s = self.nodes[inds]
        l1s = self.nodes[inds+1]

        ls = self.newton(cdf_data, inds_left, zs_cdf, l0s, l1s)
        return ls
