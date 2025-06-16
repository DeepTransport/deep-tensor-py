import abc 
from typing import Tuple
import warnings

import torch
from torch import Tensor

from ..constants import EPS


class CDF1D(abc.ABC):
    """Parent class used for evaluating the CDF and inverse CDF of all 
    one-dimensional bases.
    """

    def __init__(
        self, 
        error_tol: float = 1e-10, 
        num_newton: int = 100,
        num_regula_falsi: int = 100
    ):
        self.error_tol = error_tol
        self.num_newton = num_newton
        self.num_regula_falsi = num_regula_falsi
        return
    
    @property 
    def nodes(self) -> Tensor:
        return self._nodes 

    @nodes.setter 
    def nodes(self, value: Tensor) -> None:
        self._nodes = value 
        return
    
    @property 
    @abc.abstractmethod 
    def cardinality(self) -> Tensor:
        """The number of nodes associated with the polynomial basis of
        the CDF.
        """
        pass
    
    @property
    @abc.abstractmethod
    def domain(self) -> Tensor:
        """The domain on which polynomials used to form the CDF are 
        defined.
        """
        pass 

    @abc.abstractmethod
    def invert_cdf(self, ps: Tensor, zs: Tensor) -> Tensor:
        """Evaluates the inverse of the CDF of the target PDF at a 
        given set of values, by solving a set of root-finding problem 
        using Newton's method. If Newton's method does not converge, 
        the Regula Falsi method is applied.
        
        Parameters
        ----------
        ps: 
            An m * n matrix containing the values of the (unnormalised) 
            Radon-Nikodym derivative of the target measure with respect 
            to the weight measure, evaluated at each of the nodes of 
            the basis for the current CDF. There are two possible 
            cases: the matrix contains a single column (if the PDF is 
            the same for all zs) or a number of columns equal to the 
            number of elements of zs (if the PDF is different for all 
            zs).
        zs:
            An n-dimensional vector containing points in the interval 
            [0, 1].

        Returns
        -------
        ls:
            An n-dimensional vector containing the points in the local 
            domain corresponding to the evaluation of the inverse of 
            the CDF at each point in zs.

        """
        pass
        
    @abc.abstractmethod
    def eval_cdf(self, ps: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the CDF of the approximation to the target density 
        at a given set of values in the local domain.
        
        Parameters
        ----------
        ps:
            An m * n matrix containing the values of the (unnormalised) 
            Radon-Nikodym derivative of the target measure with respect 
            to the weight measure, evaluated at each of the nodes of 
            the basis for the current CDF. There are two possible 
            cases: the matrix contains a single column (if the PDF is 
            the same for all zs) or a number of columns equal to the 
            number of elements of zs (if the PDF is different for all 
            zs).
        ls:
            An n-dimensional vector of values in the local domain at 
            which to evaluate the CDF.

        Returns
        -------
        zs:
            An n-dimensional vector containing the values of the CDF 
            corresponding to each value of ls.
        
        """
        pass
    
    @abc.abstractmethod
    def eval_int_deriv(self, ps: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the integral of the product of the PDF and the 
        weighting function from the left-hand boundary of the local 
        domain to each element of ls.
        
        Parameters
        ----------
        ps:
            An m * n matrix containing the values of the (unnormalised) 
            Radon-Nikodym derivative of the target measure with respect 
            to the weight measure, evaluated at each of the nodes of 
            the basis for the current CDF. There are two possible 
            cases: the matrix contains a single column (if the PDF is 
            the same for all zs) or a number of columns equal to the 
            number of elements of zs (if the PDF is different for all 
            zs).
        ls: 
            An n-dimensional vector containing a set of samples from 
            the local domain.

        Returns
        -------
        zs:
            An n-dimensional vector containing the integral of the PDF
            and the product of the weighting function between the 
            left-hand boundary of the local domain and the 
            corresponding element of ls.
        
        TODO: fix this; sometimes the weighting function isn't actually 
        used.

        """
        pass
    
    @staticmethod
    def check_pdf_positive(ps: Tensor) -> None:
        """Checks whether a set of evaluations of the target PDF are 
        positive.
        """
        if (n_violations := (ps < -EPS).sum()) > 0:
            msg = (f"{n_violations} negative PDF values found. "
                   + f"Minimum value: {ps.min()}")
            warnings.warn(msg)
        return

    @staticmethod
    def check_initial_intervals(z0s: Tensor, z1s: Tensor) -> None:
        """Checks whether the function values at each side of the 
        initial interval of a rootfinding method have different signs.

        Parameters
        ----------
        z0s:
            An n-dimensional vector containing the values of the 
            function evaluated at the left-hand side of the interval.
        z1s:
            An n-dimensional vector containing the values of the 
            function evaluated at the right-hand side of the interval.
        
        Returns
        -------
        None

        """
        if (n_violations := (z0s * z1s > EPS).sum()) > 0:
            msg = (f"Rootfinding: {n_violations} initial intervals "
                   + "without roots found.")
            warnings.warn(msg)
        return
    
    def check_pdf_dims(self, ps: Tensor, xs: Tensor) -> None:
        """Checks whether the dimensions of the evaluation of the 
        target PDF(s) on the nodes of the basis of the CDF are 
        correct.
        """

        if ps.ndim != 2:
            msg = "Input PDF must be two-dimensional."
            raise Exception(msg)
        
        n_k, n_ps = ps.shape

        if n_k != self.cardinality:
            msg = ("Number of rows of PDF matrix must be equal to " 
                   + "cardinality of polynomial basis for CDF.")
            raise Exception(msg)
        
        if n_ps > 1 and n_ps != xs.numel():
            msg = ("Number of columns of PDF matrix must be equal to "
                   + "one or number of samples.")
            raise Exception(msg)
        
        return
    
    @staticmethod
    def _regula_falsi_step(
        z0s: Tensor, 
        z1s: Tensor,
        l0s: Tensor, 
        l1s: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Carries out a single regula falsi iteration."""
        dls = -z1s * (l1s - l0s) / (z1s - z0s)
        dls[dls.isinf() | dls.isnan()] = 0.0
        ls = l1s + dls
        ls = torch.clamp(ls, l0s, l1s)
        return ls, dls
    
    @staticmethod
    def _newton_step(
        ls: Tensor,
        zs: Tensor,
        dzs: Tensor,
        l0s: Tensor, 
        l1s: Tensor 
    ) -> Tuple[Tensor, Tensor]:
        """Carries out a single Newton iteration."""
        dls = -zs / dzs 
        dls[dls.isinf() | dls.isnan()] = 0.0
        ls = ls + dls 
        ls = torch.clamp(ls, l0s, l1s)
        return ls, dls
    
    def converged(self, fs: Tensor, dls: Tensor) -> bool:
        """Returns a boolean that indicates whether a rootfinding 
        method has converged.

        Parameters
        ----------
        fs:
            An n-dimensional vector containing the current values of 
            the functions we are aiming to find roots of.
        
        dls:
            An n-dimensional vector containing the steps taken at the 
            previous stage of the rootfinding method being used to find 
            the roots.
        
        Returns
        ------
        converged:
            A boolean that indicates whether the maximum absolute size 
            of the function values or stepsize values is less than the 
            error tolerance.
        
        """
        error_fs = fs.abs()
        error_dls = dls.abs()
        return float(torch.min(error_fs, error_dls).max()) < self.error_tol
    
    def print_unconverged(self, fs: Tensor, dls: Tensor, method: str) -> None:
        
        error_fs = fs.abs()
        error_dls = dls.abs()
        unconverged = (torch.min(error_fs, error_dls) >= self.error_tol)
        max_residual = error_fs.abs().max()
        
        msg = (f"{method} did not converge "
               + f"({unconverged.sum()} unconverged samples). "
               + f"Maximum residual: {max_residual:.4e}.")
        warnings.warn(msg)

        return None