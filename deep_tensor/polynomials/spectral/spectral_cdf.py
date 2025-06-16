import abc
from typing import Tuple

from torch import Tensor

from ..cdf_1d import CDF1D
from ...tools import check_finite


class SpectralCDF(CDF1D, abc.ABC):

    def __init__(self, **kwargs):
        CDF1D.__init__(self, **kwargs)
        n_sampling_nodes = 2 * self.cardinality
        self.sampling_nodes = self.grid_measure(n_sampling_nodes)
        self.cdf_basis2node = self.eval_int_basis(self.sampling_nodes)
        return
    
    @property 
    @abc.abstractmethod 
    def node2basis(self) -> Tensor:
        pass

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
        pass

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
        int_ps:
            An array of the results. Each row contains the values of 
            the indefinite integrals for each basis function for a 
            single value of ls.

        """
        pass
        
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
        pass
    
    def get_left_inds(
        self, 
        cdf_poly_nodes: Tensor, 
        zs_unnorm: Tensor
    ) -> Tensor:
        left_inds = cdf_poly_nodes >= zs_unnorm 
        left_inds[-1, :] = True
        left_inds = left_inds.int().argmax(axis=0) - 1
        left_inds = left_inds.clamp(0, self.sampling_nodes.numel() - 2)
        return left_inds
    
    def eval_int(self, coefs: Tensor, ls: Tensor) -> Tensor:
        """Returns the value of the integral of the polynomial basis 
        for the CDF (with respect to the weight function) at a set of 
        points.
        """
        int_ps = self.eval_int_basis(ls)
        fs = (int_ps * coefs.T).sum(dim=1)
        return fs
    
    def eval_int_diff(
        self,
        coefs: Tensor, 
        cdf_poly_base: Tensor,
        zs_cdf: Tensor,
        ls: Tensor
    ) -> Tensor:
        """Returns the differences between the (unnormalised) values of 
        the CDF, and the target values of the CDF, at a set of points.
        """
        dzs = self.eval_int(coefs, ls) - cdf_poly_base - zs_cdf
        return dzs

    def eval_int_newton(
        self, 
        coef: Tensor, 
        cdf_poly_base: Tensor, 
        zs_cdf: Tensor, 
        ls: Tensor
    ) -> Tuple[Tensor, Tensor]: 
        
        int_ps, ps = self.eval_int_basis_newton(ls)
        check_finite(int_ps)
        check_finite(ps)

        zs = (int_ps * coef.T).sum(dim=1)
        dzdls = (ps * coef.T).sum(dim=1)
        
        dzs = zs - cdf_poly_base - zs_cdf
        return dzs, dzdls
    
    def eval_cdf(self, ps: Tensor, ls: Tensor) -> Tensor:

        self.check_pdf_positive(ps)
        self.check_pdf_dims(ps, ls)
        
        coef = self.node2basis @ ps

        # Compute value of CDF at leftmost node and normalising constant
        poly_base = self.cdf_basis2node[0] @ coef
        poly_norm = self.cdf_basis2node[-1] @ coef - poly_base

        zs = (self.eval_int(coef, ls) - poly_base) / poly_norm
        zs = zs.clamp(0.0, 1.0)
        return zs

    def eval_int_deriv(self, ps: Tensor, ls: Tensor) -> Tensor:
        coef = self.node2basis @ ps 
        poly_base = self.cdf_basis2node[0] @ coef
        zs = self.eval_int(coef, ls) - poly_base
        return zs
    
    def newton(
        self,
        coefs: Tensor, 
        cdf_poly_base: Tensor, 
        zs_unnorm: Tensor,
        l0s: Tensor,
        l1s: Tensor
    ) -> Tensor:
        
        z0s = self.eval_int_diff(coefs, cdf_poly_base, zs_unnorm, l0s)
        z1s = self.eval_int_diff(coefs, cdf_poly_base, zs_unnorm, l1s)
        self.check_initial_intervals(z0s, z1s)

        ls, dls = self._regula_falsi_step(z0s, z1s, l0s, l1s)

        for _ in range(self.num_newton):  
            zs, dzs = self.eval_int_newton(coefs, cdf_poly_base, zs_unnorm, ls)
            ls, dls = self._newton_step(ls, zs, dzs, l0s, l1s)
            if self.converged(zs, dls):
                return ls
        
        # self.print_unconverged(zs, dls, "Newton's method")
        return self.regula_falsi(coefs, cdf_poly_base, zs_unnorm, l0s, l1s)
    
    def regula_falsi(
        self, 
        coefs: Tensor,
        cdf_poly_base: Tensor,
        zs_cdf: Tensor, 
        l0s: Tensor, 
        l1s: Tensor
    ) -> Tensor:
        
        z0s = self.eval_int_diff(coefs, cdf_poly_base, zs_cdf, l0s)
        z1s = self.eval_int_diff(coefs, cdf_poly_base, zs_cdf, l1s)
        self.check_initial_intervals(z0s, z1s)

        for _ in range(self.num_regula_falsi):

            ls, dls = self._regula_falsi_step(z0s, z1s, l0s, l1s)
            zs = self.eval_int_diff(coefs, cdf_poly_base, zs_cdf, ls)
            if self.converged(zs, dls):
                return ls 

            # Update intervals (note: the CDF is monotone increasing)
            l0s[zs < 0] = ls[zs < 0]
            l1s[zs > 0] = ls[zs > 0]
            z0s[zs < 0] = zs[zs < 0]
            z1s[zs > 0] = zs[zs > 0]
            
        self.print_unconverged(zs, dls, "Regula falsi")
        return ls
    
    def invert_cdf(self, ps: Tensor, zs: Tensor) -> Tensor:
        
        self.check_pdf_positive(ps)
        self.check_pdf_dims(ps, zs)
        
        # Compute coefficients of each basis function for each PDF
        coefs = self.node2basis @ ps

        # Evaluate sum of integrals of each basis function at each 
        # point on each grid for each PDF
        cdf_poly_nodes = self.cdf_basis2node @ coefs
        cdf_poly_base = cdf_poly_nodes[0]
        cdf_poly_nodes = cdf_poly_nodes - cdf_poly_base
        cdf_poly_norm = cdf_poly_nodes[-1]
        
        zs_cdf = zs * cdf_poly_norm
        left_inds = self.get_left_inds(cdf_poly_nodes, zs_cdf)
        l0s = self.sampling_nodes[left_inds]
        l1s = self.sampling_nodes[left_inds+1]
        
        ls = self.newton(coefs, cdf_poly_base, zs_cdf, l0s, l1s)
        return ls