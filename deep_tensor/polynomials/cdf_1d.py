import abc 
import warnings

import torch

from ..constants import EPS


class CDF1D(abc.ABC):

    def __init__(self, error_tol: float=1e-10, num_newton: int=10):
        """Parent class used for evaluating the CDF and inverse CDF of 
        all one-dimensional bases.
        """
        self.error_tol = error_tol
        self.num_newton = num_newton
        return
    
    @property 
    @abc.abstractmethod 
    def nodes(self) -> torch.Tensor: 
        """TODO"""
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
    
    def _check_initial_intervals(
        self, 
        f0s: torch.Tensor, 
        f1s: torch.Tensor 
    ) -> None:
        """Checks whether the function values at each side of the 
        initial interval of a rootfinding method have different signs.
        """
        if (num_violations := torch.sum((f0s * f1s) > 0)) > 0:
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

    def check_pdf_positive(self, pdf: torch.Tensor) -> None:
        """Verifies whether a set of evaluations of the target PDF are 
        positive.
        """
        if (num_neg_pdfs := torch.sum(pdf < -EPS)) > 0:
            msg = f"{num_neg_pdfs} negative PDF values found."
            warnings.warn(msg)
        return