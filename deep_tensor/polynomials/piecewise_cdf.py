import abc
from typing import Tuple
import warnings

import torch

from .oned_cdf import OnedCDF
from .cdf_data import CDFData
from ..constants import EPS
from ..tools import reshape_matlab


class PiecewiseCDF(OnedCDF, abc.ABC):
    """TODO: write."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @abc.abstractmethod
    def eval_int_lag_local(
        self,
        data: CDFData, 
        e_ks: torch.Tensor,
        mask: torch.Tensor,
        xs: torch.Tensor
    ):
        """TODO: write docstring etc"""
        return

    @abc.abstractmethod
    def eval_int_lag_local_deriv(
        self,
        data: CDFData, 
        e_ks: torch.Tensor,
        mask: torch.Tensor,
        xs: torch.Tensor
    ):
        return
        
    @abc.abstractmethod
    def pdf2cdf(self, pdf_vals: torch.Tensor) -> CDFData:
        return
        
    def eval_int_lag(
        self, 
        data: CDFData, 
        rs: torch.Tensor
    ):

        if data.num_samples > 1 and data.num_samples != len(rs):
            raise Exception("Data mismatch.")

        zs = torch.zeros_like(rs)

        # Indices of the gridpoint immediately to the left of each r
        e_ks = torch.sum(self.grid < rs[:, None], dim=1) - 1

        zs[e_ks == self.num_elems] = data.poly_norm[e_ks == self.num_elems]

        # TODO: figure out how to simplify this
        in_domain = ~((e_ks == -1) & (e_ks == self.num_elems))

        zs[in_domain] = self.eval_int_lag_local(data, e_ks[in_domain], in_domain, rs[in_domain])

        return zs
        
    def eval_cdf(self, pdf_vals: torch.Tensor, rs: torch.Tensor):
        """"""

        if torch.min(pdf_vals) < -1e-8:
            msg = "Negative values of PDF found."
            warnings.warn(msg)

        data = self.pdf2cdf(pdf_vals)
        zs = self.eval_int_lag(data, rs)

        if data.poly_norm.numel() > 1:
            zs = zs / reshape_matlab(data.poly_norm, zs.shape)
        else:
            zs = zs / data.poly_norm
        
        zs = reshape_matlab(zs, rs.shape)
        zs = torch.clamp(zs, EPS, 1-EPS)
        return zs
        
    def eval_int_deri(
        self, 
        pdf_vals: torch.Tensor, 
        rs: torch.Tensor
    ) -> torch.Tensor:
        
        data = self.pdf2cdf(pdf_vals)
        zs = self.eval_int_lag(data, rs)
        zs = torch.reshape(zs, rs.shape)
        return zs
    
    def invert_cdf(
        self, 
        pdf_vals: torch.Tensor, 
        xs_k: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write

        Parameters
        ----------
        pdf_vals: 
            TODO
        xi:
            The values of a set of samples from the reference domain 
            associated with the current coordinate.

        Returns
        -------
        TODO

        """

        self.check_pdf_positive(pdf_vals)
        data = self.pdf2cdf(pdf_vals)

        # num_x = xs_k.numel()

        rs_k = torch.zeros_like(xs_k)

        if data.num_samples == 1:
            raise Exception("Check this")
            rhs = xs_k * data.poly_norm
            ei = torch.sum((data.cdf_poly_grid < rhs).flatten(), 1).T

        else: 
            rhs = xs_k * data.poly_norm
            ei = torch.sum(data.cdf_poly_grid <= rhs.flatten(), 0) - 1

        mask_left = ei == -1
        mask_right = ei == self.num_elems - 1 # TODO: I'm not sure if this is right in the MATLAB code

        rs_k[mask_left] = self.domain[0]
        rs_k[mask_right] = self.domain[-1]

        mask_central = ~torch.bitwise_or(mask_left, mask_right)

        rs_k[mask_central] = self.invert_cdf_local(
            data, 
            ei[mask_central], 
            mask_central,  # Not sure why this is going in here.
            rhs[mask_central]
        )

        # TODO: figure out what this is doing.
        rs_k[torch.isnan(rs_k)] = 0.5 * (self.domain[0] + self.domain[1])
        rs_k[torch.isinf(rs_k)] = 0.5 * (self.domain[0] + self.domain[1])

        return rs_k

    def newton(
        self, 
        data: CDFData, 
        e_ks: torch.Tensor, 
        mask: torch.Tensor, 
        rhs: torch.Tensor, 
        x0s: torch.Tensor, 
        x1s: torch.Tensor
    ) -> torch.Tensor:

        f0s = self.eval_int_lag_local_search(data, e_ks, mask, rhs, x0s)
        f1s = self.eval_int_lag_local_search(data, e_ks, mask, rhs, x1s)
        self._check_initial_intervals(f0s, f1s)

        # Carry out the first iteration using the regula falsi method
        xs = x1s - f1s * (x1s - x0s) / (f1s - f0s)

        for _ in range(self.num_newton):  
            
            fs, dfs = self.eval_int_lag_local_newton(data, e_ks, mask, rhs, xs)
            
            dxs = -fs / dfs 
            dxs[torch.isnan(dxs)] = 0.0
            xs += dxs 
            xs = torch.clamp(xs, x0s, x1s)

            if self.converged(fs, dxs):
                return xs
        
        msg = "Newton's method did not converge. Trying regula falsi..."
        warnings.warn(msg)
        return self.regula_falsi(data, e_ks, mask, rhs, x0s, x1s)
    
    def regula_falsi(
        self, 
        data: CDFData, 
        e_ks: torch.Tensor, 
        mask: torch.Tensor, 
        rhs: torch.Tensor, 
        x0s: torch.Tensor, 
        x1s: torch.Tensor
    ) -> torch.Tensor:
        
        f0s = self.eval_int_lag_local_search(data, e_ks, mask, rhs, x0s)
        f1s = self.eval_int_lag_local_search(data, e_ks, mask, rhs, x1s)
        self._check_initial_intervals(f0s, f1s)

        for _ in range(100):  # TODO: make this an attribute

            dxs = -f1s * (x1s - x0s) / (f1s - f0s)
            dxs[torch.isnan(dxs)] = 0.0
            xs = x1s + dxs

            fs = self.eval_int_lag_local_search(data, e_ks, mask, rhs, xs)

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

    def eval_int_lag_local_search(
        self, 
        data: CDFData,
        e_ks: torch.Tensor, 
        mask: torch.Tensor, 
        rhs: torch.Tensor, 
        xs: torch.Tensor 
    ) -> torch.Tensor:
        """TODO: write docstring."""

        fs = self.eval_int_lag_local(data, e_ks, mask, xs)
        return fs - rhs
    
    def eval_int_lag_local_newton(
        self, 
        data: CDFData,
        e_ks: torch.Tensor, 
        mask: torch.Tensor, 
        rhs: torch.Tensor, 
        xs: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        fs, dfs = self.eval_int_lag_local_deriv(data, e_ks, mask, xs)
        fs -= rhs 
        return fs, dfs