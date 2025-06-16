import abc
from typing import Tuple

import torch
from torch import Tensor

from .piecewise import Piecewise
from ..cdf_1d import CDF1D
from ..cdf_data import CDFData


class PiecewiseCDF(CDF1D, Piecewise, abc.ABC):
    
    def __init__(self, **kwargs):
        CDF1D.__init__(self, **kwargs)
        return

    @abc.abstractmethod
    def eval_int_elem(
        self, 
        cdf_data: CDFData,
        inds_left: Tensor,
        ls: Tensor
    ) -> Tensor:
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
        pass

    @abc.abstractmethod
    def eval_int_elem_deriv(
        self, 
        cdf_data: CDFData,
        inds_left: Tensor,
        ls: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Evaluates the derivative of the unnormalised CDF at a given 
        set of values.
        
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
        dzdls:
            An n-dimensional vector containing the value of the 
            derivative of the CDF evaluated at each element of ls.
        
        """
        pass
        
    @abc.abstractmethod
    def pdf2cdf(self, ps: Tensor) -> CDFData:
        """Given evaluations of an (unnormalised) PDF (or set of 
        unnormalised PDFs), generates data on the corresponding CDF.

        Parameters
        ----------
        ps:
            A matrix containing the values of the (unnormalised) target 
            PDF evaluated at each of the nodes of the basis for the 
            current CDF. The matrix may contain multiple columns if 
            multiple PDFs are being evaluated.

        Returns
        -------
        cdf_data:
            A CDFData object containing information on the values of 
            the CDF corresponding to each PDF at each node of the 
            current basis.

        """
        pass
        
    def eval_int(self, cdf_data: CDFData, ls: Tensor) -> Tensor:

        if cdf_data.n_cdfs > 1 and cdf_data.n_cdfs != ls.numel():
            raise Exception("Data mismatch.")

        inds_left = torch.sum(self.grid < ls[:, None], dim=1) - 1
        inds_left = torch.clamp(inds_left, 0, self.num_elems-1)
        
        zs = self.eval_int_elem(cdf_data, inds_left, ls)
        return zs

    def eval_int_elem_diff(
        self, 
        data: CDFData,
        inds_left: Tensor, 
        zs_cdf: Tensor, 
        ls: Tensor 
    ) -> Tensor:
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
        dzs = self.eval_int_elem(data, inds_left, ls) - zs_cdf
        return dzs
    
    def eval_int_elem_newton(
        self, 
        cdf_data: CDFData,
        inds_left: Tensor, 
        zs_cdf: Tensor, 
        ls: Tensor 
    ) -> Tuple[Tensor, Tensor]:
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
        zs, gradzs = self.eval_int_elem_deriv(cdf_data, inds_left, ls)
        dzs = zs - zs_cdf
        return dzs, gradzs

    def eval_cdf(self, ps: Tensor, ls: Tensor) -> Tensor:

        self.check_pdf_positive(ps)
        cdf_data = self.pdf2cdf(ps)

        zs = self.eval_int(cdf_data, ls) / cdf_data.poly_norm
        zs = torch.clamp(zs, 0.0, 1.0)
        return zs
    
    def newton(
        self, 
        cdf_data: CDFData, 
        inds_left: Tensor, 
        zs_cdf: Tensor, 
        l0s: Tensor, 
        l1s: Tensor
    ) -> Tensor:
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

        z0s = self.eval_int_elem_diff(cdf_data, inds_left, zs_cdf, l0s)
        z1s = self.eval_int_elem_diff(cdf_data, inds_left, zs_cdf, l1s)
        self.check_initial_intervals(z0s, z1s)

        ls, dls = self._regula_falsi_step(z0s, z1s, l0s, l1s)

        for _ in range(self.num_newton):
            zs, dzs = self.eval_int_elem_newton(cdf_data, inds_left, zs_cdf, ls)
            ls, dls = self._newton_step(ls, zs, dzs, l0s, l1s)
            if self.converged(zs, dls):
                return ls
        
        # self.print_unconverged(zs, dls, "Newton's method")
        return self.regula_falsi(cdf_data, inds_left, zs_cdf, l0s, l1s)
    
    def regula_falsi(
        self, 
        cdf_data: CDFData, 
        inds_left: Tensor,
        zs_cdf: Tensor, 
        l0s: Tensor, 
        l1s: Tensor
    ) -> Tensor:
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
        
        z0s = self.eval_int_elem_diff(cdf_data, inds_left, zs_cdf, l0s)
        z1s = self.eval_int_elem_diff(cdf_data, inds_left, zs_cdf, l1s)
        self.check_initial_intervals(z0s, z1s)

        for _ in range(self.num_regula_falsi):

            ls, dls = self._regula_falsi_step(z0s, z1s, l0s, l1s)
            zs = self.eval_int_elem_diff(cdf_data, inds_left, zs_cdf, ls)
            if self.converged(zs / cdf_data.poly_norm, dls / cdf_data.poly_norm):
                return ls 

            # Note that the CDF is monotone increasing
            l0s[zs < 0] = ls[zs < 0]
            l1s[zs > 0] = ls[zs > 0]
            z0s[zs < 0] = zs[zs < 0]
            z1s[zs > 0] = zs[zs > 0]
            
        self.print_unconverged(zs / cdf_data.poly_norm, dls / cdf_data.poly_norm, "Regula falsi")
        return ls
    
    def eval_int_deriv(self, ps: Tensor, ls: Tensor) -> Tensor:
        
        if ps.ndim == 1:
            ps = ps[:, None]
        self.check_pdf_dims(ps, ls)
        
        cdf_data = self.pdf2cdf(ps)
        zs = self.eval_int(cdf_data, ls)
        return zs
    
    def invert_cdf_elem(
        self, 
        cdf_data: CDFData, 
        inds_left: Tensor,
        zs_cdf: Tensor
    ) -> Tensor:
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

    def invert_cdf(self, ps: Tensor, zs: Tensor) -> Tensor:

        self.check_pdf_positive(ps)
        cdf_data = self.pdf2cdf(ps)
        ls = torch.zeros_like(zs)

        zs_cdf = zs * cdf_data.poly_norm
        inds_left = (cdf_data.cdf_poly_grid <= zs_cdf).sum(dim=0) - 1
        inds_left = torch.clamp(inds_left, 0, self.num_elems-1)

        ls = self.invert_cdf_elem(cdf_data, inds_left, zs_cdf)
        return ls