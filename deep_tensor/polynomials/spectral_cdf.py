import abc
from typing import Tuple
import warnings

import torch

from .cdf_1d import CDF1D


class SpectralCDF(CDF1D, abc.ABC):

    def __init__(self, **kwargs):
        """CDF class for spectral polynomials.

        Parameters
        ----------
        TODO (??)
        
        """

        CDF1D.__init__(self, **kwargs)
        num_sampling_nodes = max(2*self.cardinality, 200)
        self.sampling_nodes = self.grid_measure(num_sampling_nodes)
        self.cdf_basis2node = self.eval_int_basis(self.sampling_nodes)
        return

    @property
    @abc.abstractmethod  # TODO: figure out whether this should go in OnedCDF or SpectralCDF.
    def cardinality(self) -> torch.Tensor:
        return
    
    @property 
    @abc.abstractmethod 
    def node2basis(self) -> torch.Tensor:
        return

    @abc.abstractmethod
    def grid_measure(self, n: int) -> torch.Tensor:
        """Returns the domain of the measure discretised on a grid of
        n points.
        
        Parameters
        ----------
        n:
            Number of discretisation points.

        Returns
        -------
        :
            The discretised domain.
        
        """
        return

    @abc.abstractmethod
    def eval_int_basis(self, xs: torch.Tensor) -> torch.Tensor:
        """Computes the indefinite integral of the product of each
        basis function and the weight function at a set of points on 
        the interval [-1, 1].

        Parameters
        ----------
        xs: 
            The set of points at which to evaluate the indefinite 
            integrals of the product of the basis function and the 
            weights.
        
        Returns
        -------
        :
            An array of the results. Each row contains the values of 
            the indefinite integrals for each basis function for a 
            single value of x.

        References
        ----------
        Cui et al. (2023), Appendix A.

        """
        return
        
    @abc.abstractmethod
    def eval_int_basis_newton(
        self, 
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the indefinite integral of the product of each 
        basis function and the weight function, and the product of the
        derivative of this integral with the weight function, at a set 
        of points on the interval [-1, 1]. 
        
        Parameters
        ----------
        xs: 
            The set of points at which to evaluate the indefinite 
            integrals of the product of the basis function and the 
            weights.
        
        Returns
        -------
        :
            An array of the integrals, and an array of the derivatives. 
            Each row contains the values of the indefinite integrals /
            derivatives for each basis function, at a single value of x.
        
        """
        return

    def update_sampling_nodes(
        self, 
        sampling_nodes: torch.Tensor
    ) -> None:
        
        self.sampling_nodes = sampling_nodes 
        self.cdf_basis2node = self.eval_int_basis(self.sampling_nodes)
        return
    
    def eval_int(
        self, 
        coef: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        basis_vals = self.eval_int_basis(x)

        if coef.shape[1] == 1:
            f = (basis_vals @ coef).flatten()
            return f

        if coef.shape[1] != x.numel():
            raise Exception("Dimension mismatch.")

        # TODO: check dimension of sum
        f = torch.sum(basis_vals * coef.T, 1)
        return f
    
    def eval_int_search(
        self,
        coef: torch.Tensor, 
        cdf_poly_base,
        rhs: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        f = self.eval_int(coef, x)
        f = f - cdf_poly_base - rhs
        return f

    def eval_int_newton(
        self, 
        coef: torch.Tensor, 
        cdf_poly_base, 
        rhs, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        
        basis_vals, deriv_vals = self.eval_int_basis_newton(x)

        fs = torch.sum(basis_vals * coef.T, 1)
        dfs = torch.sum(deriv_vals * coef.T, 1)

        fs = fs - cdf_poly_base - rhs
        return fs, dfs
    
    def eval_cdf(self, pdf: torch.Tensor, r: torch.Tensor):
        """I think pk is the PDF,
        and r is a vector of points at which to evaluate the CDF.
        
        Returns 
        -------
        :
            The values of the CDF evaluated at each point.
        
        """

        self.check_pdf_positive(pdf)

        if pdf.dim() == 1:
            pdf = pdf[:, None]

        if pdf.shape[1] > 1 and pdf.shape[1] != r.numel():
            raise Exception("Dimension mismatch.")
        
        coef = self.node2basis @ pdf
        poly_base = self.cdf_basis2node[0] @ coef
        # Normalising constant
        poly_norm = (self.cdf_basis2node[-1] - self.cdf_basis2node[0]) @ coef

        mask_left = r < self.sampling_nodes[0]
        mask_right = r > self.sampling_nodes[-1]
        mask_inside = ~(mask_left | mask_right)

        zs = torch.zeros_like(r)

        if torch.any(mask_inside):
            if pdf.shape[1] == 1:
                zs[mask_inside] = self.eval_int(coef, r[mask_inside]) - poly_base
            else:
                tmp = self.eval_int(coef[:, mask_inside], r[mask_inside])
                zs[mask_inside] = tmp.flatten() - poly_base[mask_inside].flatten()
        
        if torch.any(mask_right):
            if pdf.shape[1] == 1:
                zs[mask_right] = poly_norm 
            else:
                zs[mask_right] = poly_norm[mask_right]

        zs = zs / poly_norm.flatten()

        # z(isnan(z)) = eps;
        # z(isinf(z)) = 1-eps;
        # z(z>(1-eps)) = 1-eps;
        # z(z<eps) = eps;
    
        return zs

    def eval_int_deriv(
        self, 
        pk: torch.Tensor, 
        r: torch.Tensor
    ) -> torch.Tensor:
        """REWRITE"""
        
        coef = self.node2basis @ pk 
        base = self.cdf_basis2node[0] @ coef

        if pk.shape[1] == 1:
            z = self.eval_int(self, coef, r) - base 
        else:
            tmp = self.eval_int(coef, r)
            z = tmp - base
        
        return z
    
    def invert_cdf(
        self, 
        pdf: torch.Tensor, 
        zs_k: torch.Tensor
    ) -> torch.Tensor:
        """REWRITE"""

        self.check_pdf_positive(pdf)

        if pdf.dim() == 1:
            pdf = pdf[:, None]

        if pdf.shape[1] > 1 and pdf.shape[1] != zs_k.numel():
            raise Exception("Dimension mismatch.")
        
        coefs = self.node2basis @ pdf
        cdf_poly_nodes = self.cdf_basis2node @ coefs
        cdf_poly_base = cdf_poly_nodes[0]
        cdf_poly_nodes = cdf_poly_nodes - cdf_poly_base
        cdf_poly_norm = cdf_poly_nodes[-1]

        rs_k = torch.zeros_like(zs_k)

        rhs = zs_k * cdf_poly_norm  # vertical??
        left_inds = torch.sum(cdf_poly_nodes < rhs, 0).int() - 1
        
        mask_left = left_inds == -1
        mask_right = left_inds == (self.sampling_nodes.numel() - 1)

        rs_k[mask_left] = self.sampling_nodes[0]
        rs_k[mask_right] = self.sampling_nodes[-1]

        mask_central = ~torch.bitwise_or(mask_left, mask_right)
        
        x0s = self.sampling_nodes[left_inds[mask_central]]
        x1s = self.sampling_nodes[left_inds[mask_central]+1]
        
        rs_k[mask_central] = self.newton(
            coefs[:, mask_central], 
            cdf_poly_base[mask_central],
            rhs[mask_central], 
            x0s, 
            x1s
        )
        
        rs_k[torch.isnan(rs_k)] = 0.5 * (self.domain[0] + self.domain[1])
        rs_k[torch.isinf(rs_k)] = 0.5 * (self.domain[0] + self.domain[1])
        
        return rs_k

    def newton(
        self,
        coefs: torch.Tensor, 
        cdf_poly_base: torch.Tensor, 
        rhs: torch.Tensor,
        x0s: torch.Tensor,
        x1s: torch.Tensor
    ) -> torch.Tensor:
        
        f0s = self.eval_int_search(coefs, cdf_poly_base, rhs, x0s)
        f1s = self.eval_int_search(coefs, cdf_poly_base, rhs, x1s)
        self._check_initial_intervals(f0s, f1s)

        # Carry out the first iteration using the regula falsi method
        xs = x1s - f1s * (x1s - x0s) / (f1s - f0s)

        for _ in range(self.num_newton):  
            
            fs, dfs = self.eval_int_newton(coefs, cdf_poly_base, rhs, xs)
            
            dxs = -fs / dfs 
            dxs[torch.isnan(dxs)] = 0.0
            xs += dxs 
            xs = torch.clamp(xs, x0s, x1s)

            if self.converged(fs, dxs):
                return xs
        
        msg = "Newton's method did not converge. Trying regula falsi..."
        warnings.warn(msg)
        return self.regula_falsi(coefs, cdf_poly_base, rhs, x0s, x1s)
    
    def regula_falsi(
        self, 
        coefs: torch.Tensor,
        cdf_poly_base: torch.Tensor,
        rhs: torch.Tensor, 
        x0s: torch.Tensor, 
        x1s: torch.Tensor
    ) -> torch.Tensor:
        
        f0s = self.eval_int_search(coefs, cdf_poly_base, rhs, x0s)
        f1s = self.eval_int_search(coefs, cdf_poly_base, rhs, x1s)
        self._check_initial_intervals(f0s, f1s)

        for _ in range(100):  # TODO: make this an attribute

            dxs = -f1s * (x1s - x0s) / (f1s - f0s)
            dxs[torch.isnan(dxs)] = 0.0
            xs = x1s + dxs

            fs = self.eval_int_search(coefs, cdf_poly_base, rhs, xs)

            if self.converged(fs, dxs):
                return xs 

            # Update intervals (note: the CDF is monotone increasing)
            x0s[fs < 0] = xs[fs < 0]
            x1s[fs > 0] = xs[fs > 0]
            f0s[fs < 0] = fs[fs < 0]
            f1s[fs > 0] = fs[fs > 0]
            
        msg = "Regula falsi did not converge in 100 iterations."
        warnings.warn(msg)
        return xs
       
