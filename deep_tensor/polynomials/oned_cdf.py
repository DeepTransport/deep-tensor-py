import abc 
import warnings

import torch

from .cdf_data import CDFData
from ..constants import EPS


class OnedCDF(abc.ABC):
    """Parent class used for evaluating the CDF and inverse CDF of all
    one-dimensional bases.
    """

    def __init__(self, error_tol: float=1e-10, num_newton: int=10):
        self.error_tol = error_tol
        self.num_newton = num_newton
        return
    
    @property 
    @abc.abstractmethod 
    def nodes(self) -> torch.Tensor: 
        """TODO: write description."""
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
        """TODO: write description."""
        return 

    @abc.abstractmethod
    def invert_cdf(self):
        """Inverts the CDF by solving a root-finding problem using 
        Newton's method. If Newton's method does not converge within
        10 iterations, the Regula Falsi method is applied.
        """
        return
        
    @abc.abstractmethod
    def eval_cdf(
        self, 
        pdf: torch.Tensor, 
        r: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the CDF."""
        return
    
    @abc.abstractmethod
    def eval_int_deriv(self):
        """??"""
        return
    
    @abc.abstractmethod
    def newton(
        self,
        data: CDFData, 
        e_ks: torch.Tensor, 
        mask: torch.Tensor, 
        rhs: torch.Tensor, 
        x0s: torch.Tensor, 
        x1s: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return
    
    @abc.abstractmethod
    def regula_falsi(
        self,
        data: CDFData, 
        e_ks: torch.Tensor, 
        mask: torch.Tensor, 
        rhs: torch.Tensor, 
        x0s: torch.Tensor, 
        x1s: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""
        return
    
    def _check_initial_intervals(
        self, 
        f0s: torch.Tensor, 
        f1s: torch.Tensor 
    ) -> None:
        """Checks whether the function values at each side of the 
        initial interval of a rootfinding method have different signs.
        """

        if (num_violations := torch.sum((f0s * f1s) > 0)) == 0:
            return

        print(torch.min(f1s-f0s))

        print(torch.min(f0s))
        print(torch.max(f0s))
        print(torch.min(f1s))
        print(torch.max(f1s))

        msg = (f"Rootfinding: {num_violations} initial intervals "
               + "without roots found.")
        warnings.warn(msg)
        return
    
    def converged(self, fs, dxs):
        """Returns a boolean that indicates whether a rootfinding 
        method has converged.
        """

        error_f = torch.max(torch.abs(fs))
        error_dx = torch.max(torch.abs(dxs))

        return torch.min(error_f, error_dx) < self.error_tol
    
    # @abc.abstractmethod
    # def eval_cdf_deriv(self):
    #     """Evaluates the derivative of the conditional CDF. This 
    #     function is used to compute the Jacobian of the inverse 
    #     Rosenblatt transport.
    #     """
    #     return
    # TODO: figure out whether this should be here or not.

    def check_pdf_positive(self, pdf: torch.Tensor) -> None:
        
        if (num_neg_pdfs := torch.sum(pdf < -EPS)) > 0:
            msg = f"{num_neg_pdfs} negative PDF values found."
            warnings.warn(msg)