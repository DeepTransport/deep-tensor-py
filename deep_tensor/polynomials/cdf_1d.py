import abc 
import warnings

import torch

from ..constants import EPS


class CDF1D(abc.ABC):

    def __init__(
        self, 
        error_tol: float=1e-10, 
        num_newton: int=10,
        num_regula_falsi: int=100
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
    def nodes(self) -> torch.Tensor: 
        """The nodes associated with the CDF."""
        return 
    
    @property 
    @abc.abstractmethod 
    def cardinality(self) -> torch.Tensor:
        """The number of nodes associated with the polynomial basis of
        the CDF.
        """
        return
    
    @property
    @abc.abstractmethod
    def domain(self) -> torch.Tensor:
        """The domain on which polynomials used to form the CDF are 
        defined.
        """
        return 

    # @abc.abstractmethod 
    # def invert_cdf_local(self):
    #     """TODO: write docstring"""
    #     return

    @abc.abstractmethod
    def invert_cdf(
        self, 
        pls: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the inverse of the CDF of the target PDF at a 
        given set of values, by solving a set of root-finding problem 
        using Newton's method. If Newton's method does not converge, 
        the Regula Falsi method is applied.
        
        Parameters
        ----------
        pls: 
            A matrix containing the values of the target PDF evaluated 
            at each of the nodes of the basis for the current CDF.
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
    def eval_cdf(
        self, 
        pls: torch.Tensor, 
        ls: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the CDF of the approximation to the target density 
        at a given set of values in the local domain.
        
        Parameters
        ----------
        pls:
            The values of the (unnormalised) approximation to the 
            target density function evaluated at each of the nodes of 
            the polynomial basis of the CDF.
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
    def eval_int_deriv(self):
        """??"""
        return
    
    def check_pdf_positive(self, pdf: torch.Tensor) -> None:
        """Verifies whether a set of evaluations of the target PDF are 
        positive.
        """
        if torch.sum(pdf < -EPS) > 0:
            msg = "Negative PDF values found."
            warnings.warn(msg)
        return

    def check_initial_intervals(
        self, 
        z0s: torch.Tensor, 
        z1s: torch.Tensor 
    ) -> None:
        """Checks whether the function values at each side of the 
        initial interval of a rootfinding method have different signs.
        """
        if (num_violations := torch.sum(z0s * z1s > 0)) > 0:
            msg = (f"Rootfinding: {num_violations} initial intervals "
                   + "without roots found.")
            warnings.warn(msg)
        return
    
    def converged(self, fs: torch.Tensor, dxs: torch.Tensor) -> bool:
        """Returns a boolean that indicates whether a rootfinding 
        method has converged.
        """
        error_f = fs.abs().max()
        error_dx = dxs.abs().max()
        return torch.min(error_f, error_dx) < self.error_tol