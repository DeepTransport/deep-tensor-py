import abc
from typing import Tuple
import warnings

import torch
from torch import Tensor

from ..cdf_1d import CDF1D
from ...tools import check_finite


class SpectralCDF(CDF1D, abc.ABC):

    def __init__(self, **kwargs):
        CDF1D.__init__(self, **kwargs)
        num_sampling_nodes = max(2*self.cardinality, 200)
        self.sampling_nodes = self.grid_measure(num_sampling_nodes)
        self.cdf_basis2node = self.eval_int_basis(self.sampling_nodes)
        return
    
    @property 
    @abc.abstractmethod 
    def node2basis(self) -> Tensor:
        return

    @abc.abstractmethod
    def grid_measure(self, n: int) -> Tensor:
        """Returns the domain of the measure discretised on a grid of
        n points.
        
        Parameters
        ----------
        n:
            Number of discretisation points.

        Returns
        -------
        ls:
            The discretised domain.
        
        """
        return

    @abc.abstractmethod
    def eval_int_basis(self, ls: Tensor) -> Tensor:
        """Computes the indefinite integral of the product of each
        basis function and the weight function at a set of points on 
        the interval [-1, 1].

        Parameters
        ----------
        ls: 
            The set of points at which to evaluate the indefinite 
            integrals of the product of the basis function and the 
            weights.
        
        Returns
        -------
        :
            An array of the results. Each row contains the values of 
            the indefinite integrals for each basis function for a 
            single value of ls.

        References
        ----------
        Cui et al. (2023), Appendix A.

        """
        return
        
    @abc.abstractmethod
    def eval_int_basis_newton(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the indefinite integral of the product of each 
        basis function and the weight function, and the product of the
        derivative of this integral with the weight function, at a set 
        of points on the interval [-1, 1]. 
        
        Parameters
        ----------
        ls: 
            The set of points at which to evaluate the indefinite 
            integrals of the product of the basis function and the 
            weights.
        
        Returns
        -------
        :
            An array of the integrals, and an array of the derivatives. 
            Each row contains the values of the indefinite integrals /
            derivatives for each basis function, at a single value of 
            ls.
        
        """
        return

    def update_sampling_nodes(self, sampling_nodes: Tensor) -> None:
        self.sampling_nodes = sampling_nodes 
        self.cdf_basis2node = self.eval_int_basis(self.sampling_nodes)
        return
    
    def eval_int(self, coefs: Tensor, ls: Tensor) -> Tensor:
        
        basis_vals = self.eval_int_basis(ls)

        if coefs.shape[1] == 1:
            fs = basis_vals @ coefs
            return fs.flatten()

        if coefs.shape[1] != ls.numel():
            raise Exception("Dimension mismatch.")
        
        fs = torch.sum(basis_vals * coefs.T, dim=1)
        return fs
    
    def eval_int_search(
        self,
        coefs: Tensor, 
        cdf_poly_base: Tensor,
        zs_cdf: Tensor,
        ls: Tensor
    ) -> Tensor:
        dzs = self.eval_int(coefs, ls) - cdf_poly_base - zs_cdf
        return dzs

    def eval_int_newton(
        self, 
        coef: Tensor, 
        cdf_poly_base, 
        rhs, 
        ls: Tensor
    ) -> Tuple[Tensor, Tensor]: 
        
        basis_vals, deriv_vals = self.eval_int_basis_newton(ls)

        check_finite(basis_vals)
        check_finite(deriv_vals)

        fs = torch.sum(basis_vals * coef.T, 1)
        dfs = torch.sum(deriv_vals * coef.T, 1)

        fs = fs - cdf_poly_base - rhs
        return fs, dfs
    
    def eval_cdf(self, ps: Tensor, ls: Tensor) -> Tensor:

        if ps.ndim == 1:
            ps = ps[:, None]

        self.check_pdf_positive(ps)
        self.check_pdf_dims(ps, ls)
        
        coef = self.node2basis @ ps
        poly_base = self.cdf_basis2node[0] @ coef
        # Normalising constant
        poly_norm = (self.cdf_basis2node[-1] - self.cdf_basis2node[0]) @ coef

        mask_left = ls < self.sampling_nodes[0]
        mask_right = ls > self.sampling_nodes[-1]
        mask_inside = ~(mask_left | mask_right)

        zs = torch.zeros_like(ls)

        if torch.any(mask_inside):
            if ps.shape[1] == 1:
                zs[mask_inside] = self.eval_int(coef, ls[mask_inside]) - poly_base
            else:
                tmp = self.eval_int(coef[:, mask_inside], ls[mask_inside])
                zs[mask_inside] = tmp.flatten() - poly_base[mask_inside].flatten()
        
        if torch.any(mask_right):
            if ps.shape[1] == 1:
                zs[mask_right] = poly_norm 
            else:
                zs[mask_right] = poly_norm[mask_right]

        zs = zs / poly_norm.flatten()
        return zs

    def eval_int_deriv(self, ps: Tensor, ls: Tensor) -> Tensor:
        """TODO: rewrite. zs should be renamed."""
        
        coef = self.node2basis @ ps 
        base = self.cdf_basis2node[0] @ coef

        zs = self.eval_int(coef, ls) - base
        return zs
    
    def invert_cdf(self, ps: Tensor, zs: Tensor) -> Tensor:

        if ps.ndim == 1:
            ps = ps[:, None]
        
        self.check_pdf_positive(ps)
        self.check_pdf_dims(ps, zs)
        
        coefs = self.node2basis @ ps
        cdf_poly_nodes = self.cdf_basis2node @ coefs
        cdf_poly_base = cdf_poly_nodes[0]
        cdf_poly_nodes = cdf_poly_nodes - cdf_poly_base
        cdf_poly_norm = cdf_poly_nodes[-1]

        zs_cdf = zs * cdf_poly_norm  # unnormalise
        left_inds = (cdf_poly_nodes < zs_cdf).sum(dim=0) - 1
        left_inds = left_inds.clamp(0, self.sampling_nodes.numel()-2)
        l0s = self.sampling_nodes[left_inds]
        l1s = self.sampling_nodes[left_inds+1]
        
        ls = self.newton(coefs, cdf_poly_base, zs_cdf, l0s, l1s)
        check_finite(ls)
        return ls

    def newton(
        self,
        coefs: Tensor, 
        cdf_poly_base: Tensor, 
        zs_cdf: Tensor,
        l0s: Tensor,
        l1s: Tensor
    ) -> Tensor:
        
        z0s = self.eval_int_search(coefs, cdf_poly_base, zs_cdf, l0s)
        z1s = self.eval_int_search(coefs, cdf_poly_base, zs_cdf, l1s)
        self.check_initial_intervals(z0s, z1s)

        # Carry out the first iteration using the regula falsi method
        ls = l1s - z1s * (l1s - l0s) / (z1s - z0s)

        for _ in range(self.num_newton):  
            
            zs, dzs = self.eval_int_newton(coefs, cdf_poly_base, zs_cdf, ls)
            
            dls = -zs / dzs 
            check_finite(dls)
            dls[torch.isinf(dls)] = 0.0
            ls += dls 
            ls = torch.clamp(ls, l0s, l1s)

            if self.converged(zs, dls):
                return ls
        
        msg = "Newton's method did not converge. Trying regula falsi..."
        warnings.warn(msg)
        return self.regula_falsi(coefs, cdf_poly_base, zs_cdf, l0s, l1s)
    
    def regula_falsi(
        self, 
        coefs: Tensor,
        cdf_poly_base: Tensor,
        zs_cdf: Tensor, 
        l0s: Tensor, 
        l1s: Tensor
    ) -> Tensor:
        
        z0s = self.eval_int_search(coefs, cdf_poly_base, zs_cdf, l0s)
        z1s = self.eval_int_search(coefs, cdf_poly_base, zs_cdf, l1s)
        self.check_initial_intervals(z0s, z1s)

        for _ in range(self.num_regula_falsi):

            dls = -z1s * (l1s - l0s) / (z1s - z0s)
            check_finite(dls)
            dls[torch.isinf(dls)] = 0.0
            ls = l1s + dls

            zs = self.eval_int_search(coefs, cdf_poly_base, zs_cdf, ls)

            if self.converged(zs, dls):
                return ls 

            # Update intervals (note: the CDF is monotone increasing)
            l0s[zs < 0] = ls[zs < 0]
            l1s[zs > 0] = ls[zs > 0]
            z0s[zs < 0] = zs[zs < 0]
            z1s[zs > 0] = zs[zs > 0]
            
        msg = ("Regula falsi did not converge in "
               + f"{self.num_regula_falsi} iterations.")
        warnings.warn(msg)
        return ls
       
