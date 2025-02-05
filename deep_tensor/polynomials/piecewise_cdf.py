import abc
from typing import Tuple
import warnings

import torch

from .cdf_1d import CDF1D
from .cdf_data import CDFData
from ..constants import EPS


class PiecewiseCDF(CDF1D, abc.ABC):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @abc.abstractmethod
    def eval_int_lag_local(
        self, 
        cdf_data: CDFData,
        inds_left: torch.Tensor,
        ls: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the (unnormalised) CDF at a given set of values.
        
        Parameters
        ----------
        cdf_data:
            An object containing information about the properties of 
            the CDF.
        inds_left:
            An n-dimensional vector containing the indices of the 
            points of the grid on which the target PDF is discretised 
            that are immediately to the left of each value in ls.
        ls:
            An n-dimensional vector containing a set of points in the 
            local domain at which to evaluate the CDF.

        Returns
        -------
        zs:
            An n-dimensional vector containing the value of the CDF 
            evaluated at each element of ls.
        
        """
        return

    @abc.abstractmethod
    def eval_int_lag_local_deriv(
        self, 
        cdf_data: CDFData,
        inds_left: torch.Tensor,
        ls: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return
        
    @abc.abstractmethod
    def pdf2cdf(self, pls: torch.Tensor) -> CDFData:
        """Given evaluations of an (unnormalised) PDF (or set of 
        unnormalised PDFs), generates data on the corresponding CDF.

        Parameters
        ----------
        pls:
            A matrix containing the values of the (unnormalised) target 
            PDF evaluated at each of the nodes of the basis for the 
            current CDF. The matrix may contain multiple columns if 
            multiple PDFs are under consideration.

        Returns
        -------
        cdf_data:
            A CDFData object containing information on the values of 
            the CDF corresponding to each PDF at each node of the 
            current basis.

        """
        return
        
    def eval_int_lag(
        self, 
        cdf_data: CDFData, 
        ls: torch.Tensor
    ) -> torch.Tensor:

        if cdf_data.n_cdfs > 1 and cdf_data.n_cdfs != ls.numel():
            raise Exception("Data mismatch.")

        zs = torch.zeros_like(ls)

        inds_left = torch.sum(self.grid < ls[:, None], dim=1) - 1
        inds_left = torch.clamp(inds_left, 0, self.num_elems-1)
        
        zs = self.eval_int_lag_local(cdf_data, inds_left, ls)
        return zs

    def eval_int_lag_local_search(
        self, 
        data: CDFData,
        inds_left: torch.Tensor, 
        zs_cdf: torch.Tensor, 
        ls: torch.Tensor 
    ) -> torch.Tensor:
        """Returns the difference between the values of the current 
        CDF evaluated at a set of points in the local domain and a set
        of values of the CDF we are aiming to compute the inverse of.

        Parameters
        ----------
        cdf_data:
            An object containing information about the CDF.
        inds_left:
            An n-dimensional vector containing the indices of the 
            points of the grid on which the target PDF is discretised 
            that are immediately to the left of each value in ls.
        zs_cdf:
            An n-dimensional vector of values we are aiming to evaluate 
            the inverse of the (unnormalised) CDF at.
        ls: 
            An n-dimensional vector containing a current set of 
            estimates (in the local domain) for the inverse of the CDF 
            at each value of zs_cdf.

        Returns
        -------
        dzs:
            An n-dimensional vector containing the differences between 
            the value of the CDF evaluated at each element of ls and 
            the values of zs_cdf.

        """
        dzs = self.eval_int_lag_local(data, inds_left, ls) - zs_cdf
        return dzs
    
    def eval_int_lag_local_newton(
        self, 
        cdf_data: CDFData,
        inds_left: torch.Tensor, 
        zs_cdf: torch.Tensor, 
        ls: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the difference between the values of the 
        (unnormalised) CDF evaluated at a set of points in the local 
        domain and a set of values of the CDF we are aiming to compute 
        the inverse of, as well as the gradient of the CDF.

        Parameters
        ----------
        cdf_data:
            An object containing information about the CDF.
        inds_left:
            An n-dimensional vector containing the indices of the 
            points of the grid on which the target PDF is discretised 
            that are immediately to the left of each value in ls.
        zs_cdf:
            An n-dimensional vector of values we are aiming to evaluate 
            the inverse of the (unnormalised) CDF at.
        ls: 
            An n-dimensional vector containing a current set of 
            estimates (in the local domain) for the inverse of the CDF 
            at each value of zs_cdf.

        Returns
        -------
        dzs:
            An n-dimensional vector containing the differences between 
            the value of the (unnormalised) CDF evaluated at each 
            element of ls and the values of zs_cdf.
        gradzs:
            An n-dimensional vector containing the gradient of the 
            unnormalised CDF evaluated at each element in ls.

        """
        zs, gradzs = self.eval_int_lag_local_deriv(cdf_data, inds_left, ls)
        dzs = zs - zs_cdf
        return dzs, gradzs

    def eval_cdf(
        self, 
        pls: torch.Tensor, 
        ls: torch.Tensor
    ) -> torch.Tensor:

        self.check_pdf_positive(pls)
        cdf_data = self.pdf2cdf(pls)

        zs = self.eval_int_lag(cdf_data, ls) / cdf_data.poly_norm
        zs = torch.clamp(zs, EPS, 1.0-EPS)
        return zs
    
    def newton(
        self, 
        cdf_data: CDFData, 
        inds_left: torch.Tensor, 
        zs_cdf: torch.Tensor, 
        l0s: torch.Tensor, 
        l1s: torch.Tensor
    ) -> torch.Tensor:
        """Inverts a CDF using Newton's method.
        
        Parameters
        ----------
        cdf_data:
            An object containing information about the CDF.
        inds_left:
            An n-dimensional vector containing the indices of the 
            points of the grid on which the target PDF is discretised 
            that are immediately to the left of each value in ls.
        zs_cdf:
            An n-dimensional vector containing a set of values in the 
            range [0, Z], where Z is the normalising constant 
            associated with the current target PDF.
        l0s:
            An n-dimensional vector containing the locations of the 
            nodes of the current polynomial basis (in the local domain) 
            directly to the left of each value in zs_cdf.
        l1s:
            An n-dimensional vector containing the locations of the 
            nodes of the current polynomial basis (in the local domain) 
            directly to the right of each value in zs_cdf.

        Returns
        -------
        ls:
            An n-dimensional vector containing the values (in the local
            domain) of the inverse of the CDF evaluated at each element 
            in zs_cdf.
        
        """

        z0s = self.eval_int_lag_local_search(cdf_data, inds_left, zs_cdf, l0s)
        z1s = self.eval_int_lag_local_search(cdf_data, inds_left, zs_cdf, l1s)
        self.check_initial_intervals(z0s, z1s)

        # Carry out the first iteration using the regula falsi method
        ls = l1s - z1s * (l1s - l0s) / (z1s - z0s)

        for _ in range(self.num_newton):  
            
            zs, dzs = self.eval_int_lag_local_newton(cdf_data, inds_left, zs_cdf, ls)
            
            dls = -zs / dzs 
            dls[torch.isinf(dls)] = 0.0
            dls[torch.isnan(dls)] = 0.0
            ls += dls 
            ls = torch.clamp(ls, l0s, l1s)

            if self.converged(zs, dls):
                return ls
        
        msg = "Newton's method did not converge. Trying regula falsi..."
        warnings.warn(msg)
        return self.regula_falsi(cdf_data, inds_left, zs_cdf, l0s, l1s)
    
    def regula_falsi(
        self, 
        cdf_data: CDFData, 
        inds_left: torch.Tensor,
        zs_cdf: torch.Tensor, 
        l0s: torch.Tensor, 
        l1s: torch.Tensor
    ) -> torch.Tensor:
        """Inverts a CDF using the regula falsi method.
        
        Parameters
        ----------
        cdf_data:
            An object containing information about the CDF.
        inds_left:
            An n-dimensional vector containing the indices of the 
            points of the grid on which the target PDF is discretised 
            that are immediately to the left of each value in ls.
        zs_cdf:
            An n-dimensional vector containing a set of values in the 
            range [0, Z], where Z is the normalising constant 
            associated with the current target PDF.
        l0s:
            An n-dimensional vector containing the locations of the 
            nodes of the current polynomial basis (in the local domain) 
            directly to the left of each value in zs_cdf.
        l1s:
            An n-dimensional vector containing the locations of the 
            nodes of the current polynomial basis (in the local domain) 
            directly to the right of each value in zs_cdf.

        Returns
        -------
        ls:
            An n-dimensional vector containing the values (in the local
            domain) of the inverse of the CDF evaluated at each element 
            in zs_cdf.
        
        """
        
        z0s = self.eval_int_lag_local_search(cdf_data, inds_left, zs_cdf, l0s)
        z1s = self.eval_int_lag_local_search(cdf_data, inds_left, zs_cdf, l1s)
        self.check_initial_intervals(z0s, z1s)

        for _ in range(self.num_regula_falsi):

            dls = -z1s * (l1s - l0s) / (z1s - z0s)
            dls[torch.isinf(dls)] = 0.0
            ls = l1s + dls

            zs = self.eval_int_lag_local_search(cdf_data, inds_left, zs_cdf, ls)

            if self.converged(zs, dls):
                return ls 

            # Note that the CDF is monotone increasing
            l0s[zs < 0] = ls[zs < 0]
            l1s[zs > 0] = ls[zs > 0]
            z0s[zs < 0] = zs[zs < 0]
            z1s[zs > 0] = zs[zs > 0]
            
        msg = "Regula falsi did not converge in 100 iterations."
        warnings.warn(msg)
        return ls
    
    def eval_int_deriv(
        self, 
        pls: torch.Tensor, 
        ls: torch.Tensor
    ) -> torch.Tensor:
        
        cdf_data = self.pdf2cdf(pls)
        zs = self.eval_int_lag(cdf_data, ls)
        return zs
    
    def invert_cdf_local(
        self, 
        cdf_data: CDFData, 
        inds_left: torch.Tensor,
        zs_cdf: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the inverse of the CDF corresponding to the 
        (unnormalised) target PDF at a given set of values.
        
        Parameters
        ----------
        cdf_data:
            An object containing information about the properties of 
            the CDF.
        inds_left:
            An n-dimensional vector containing the indices of the 
            points of the grid on which the target PDF is discretised 
            that are immediately to the left of each value in ls.
        zs_cdf:
            An n-dimensional vector of values in the range [0, Z], 
            where Z is the normalising constant associated with the 
            (unnormalised) target PDF.

        Returns
        -------
        ls:
            An n-dimensional vector containing the inverse of the CDF 
            evaluated at each element in zs_cdf.
        
        """
        l0s, l1s = self.grid[inds_left], self.grid[inds_left+1]
        ls = self.newton(cdf_data, inds_left, zs_cdf, l0s, l1s)
        return ls

    def invert_cdf(
        self, 
        pls: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:

        # # TEMP
        # pls = torch.zeros((8, 81))
        # pls[:, :41] = torch.linspace(0, 1, 41)
        # pls[:, 40:] = torch.linspace(1, 0, 41)
        # pls = pls.T
        # zs = torch.linspace(0.0, 1.0, 8)

        self.check_pdf_positive(pls)
        cdf_data = self.pdf2cdf(pls)
        ls = torch.zeros_like(zs)

        # print(cdf_data.poly_coef[:, :, 0])
        # print(cdf_data.poly_coef[:, :, 1])

        zs_cdf = zs * cdf_data.poly_norm
        inds_left = (cdf_data.cdf_poly_grid <= zs_cdf).sum(dim=0) - 1
        inds_left = torch.clamp(inds_left, 0, self.num_elems-1)

        ls = self.invert_cdf_local(cdf_data, inds_left, zs_cdf)
        return ls