import abc 
import warnings

import torch
from torch import Tensor

from ..constants import EPS


class CDF1D(abc.ABC):

    def __init__(
        self, 
        error_tol: float = 1e-10, 
        num_newton: int = 10,
        num_regula_falsi: int = 100
    ):
        """Parent class used for evaluating the CDF and inverse CDF of 
        all one-dimensional bases.

        """
        self.error_tol = error_tol
        self.num_newton = num_newton
        self.num_regula_falsi = num_regula_falsi
        return
    
    @property 
    @abc.abstractmethod 
    def nodes(self) -> Tensor: 
        """The nodes associated with the CDF."""
        return 
    
    @property 
    @abc.abstractmethod 
    def cardinality(self) -> Tensor:
        """The number of nodes associated with the polynomial basis of
        the CDF.

        """
        return
    
    @property
    @abc.abstractmethod
    def domain(self) -> Tensor:
        """The domain on which polynomials used to form the CDF are 
        defined.
        
        """
        return 

    # @abc.abstractmethod 
    # def invert_cdf_local(self):
    #     """TODO: write docstring"""
    #     return

    @abc.abstractmethod
    def invert_cdf(self, ps: Tensor, zs: Tensor) -> Tensor:
        """Evaluates the inverse of the CDF of the target PDF at a 
        given set of values, by solving a set of root-finding problem 
        using Newton's method. If Newton's method does not converge, 
        the Regula Falsi method is applied.
        
        Parameters
        ----------
        ps: 
            A matrix containing the values of the (unnormalised) target 
            PDF evaluated at each of the nodes of the basis for the 
            current CDF. There are two possible cases: the matrix 
            contains a single column (if the PDF is the same for all 
            zs) or a number of columns equal to the number of elements 
            of zs (if the PDF is different for all zs).
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
        return
        
    @abc.abstractmethod
    def eval_cdf(self, ps: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the CDF of the approximation to the target density 
        at a given set of values in the local domain.
        
        Parameters
        ----------
        ps:
            A matrix containing the values of the (unnormalised) target 
            PDF evaluated at each of the nodes of the basis for the 
            current CDF. There are two possible cases: the matrix 
            contains a single column (if the PDF is the same for all 
            zs) or a number of columns equal to the number of elements 
            of zs (if the PDF is different for all zs).
        ls:
            An n-dimensional vector of values in the local domain at 
            which to evaluate the CDF.

        Returns
        -------
        zs:
            An n-dimensional vector containing the values of the CDF 
            corresponding to each value of ls.
        
        """
        return
    
    @abc.abstractmethod
    def eval_int_deriv(self, ps: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the integral of the product of the PDF and the 
        weighting function from the left-hand boundary of the local 
        domain to each element of ls.
        
        Parameters
        ----------
        ps:
            An m * n matrix containing the evaluations of the target 
            PDF for each value of ls. The number of rows of ps should 
            be equal to the number of nodes of the CDF basis, and the 
            number of columns of ps should be equal to the number of 
            elements of ls.
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
        return
    
    @staticmethod
    def check_pdf_positive(ps: Tensor) -> None:
        """Verifies whether a set of evaluations of the target PDF are 
        positive.

        """
        if torch.sum(ps < -EPS) > 0:
            msg = "Negative PDF values found."
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
        if (num_violations := torch.sum(z0s * z1s > EPS)) > 0:
            msg = (f"Rootfinding: {num_violations} initial intervals "
                   + "without roots found.")
            warnings.warn(msg)
        return
    
    def check_pdf_dims(self, ps: Tensor, xs: Tensor) -> None:
        """Checks whether the dimensions of the evaluation of the 
        target PDF(s) on the nodes of the basis of the CDF are 
        correct.
        
        """
        
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
            Boolean that indicates whether the maximum absolute size of
            either the function or stepsize values is less than the 
            error tolerance.
        
        """
        error_fs = fs.abs().max()
        error_dls = dls.abs().max()
        return torch.min(error_fs, error_dls) < self.error_tol