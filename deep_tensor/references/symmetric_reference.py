import abc
from typing import Tuple

import torch

from ..domains import BoundedDomain, Domain
from .reference import Reference


DEFAULT_MU = 0
DEFAULT_SIGMA = 1
DEFAULT_DOMAIN = BoundedDomain(bounds=torch.tensor([-4.0, 4.0]))
EPSILON = 1e-8


class SymmetricReference(Reference, abc.ABC):
    
    def __init__(
        self, 
        mu: float=DEFAULT_MU, 
        sigma: float=DEFAULT_SIGMA,
        domain: Domain=DEFAULT_DOMAIN
    ):
        
        self.mu = mu
        self.sigma = sigma
        self.domain = domain
        self.is_truncated = isinstance(domain, BoundedDomain)
        self.set_cdf_bounds()
        return

    def set_cdf_bounds(self):
        
        if self.is_truncated:
            self.left = self.eval_ref_cdf((self.domain.left-self.mu)/self.sigma)[0]
            self.right = self.eval_ref_cdf((self.domain.right-self.mu)/self.sigma)[0]
        else:
            self.left = 0.0
            self.right = 1.0
        return

    @abc.abstractmethod
    def eval_ref_cdf(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the values of the CDF and PDF of the unit reference
        distribution evaluated at each value of z.
        
        Parameters
        ----------
        z: 
            The values at which to evaluate the CDF and PDF of the 
            unit reference distribution.
        
        Returns
        -------
        :
            The CDF and PDF of the reference distribution evaluated at
            each value of z.
        
        """
        return
    
    @abc.abstractmethod
    def eval_ref_pdf(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the values of the PDF and gradient of the PDF of the 
        unit reference distribution evaluated at each value of z.
        
        Parameters
        ----------
        z: 
            The values at which to evaluate the PDF and gradient of the 
            PDF of the unit reference distribution.
        
        Returns
        -------
        :
            The PDF and gradient of the PDF of the reference 
            distribution evaluated at each value of z.
        
        """
        return

    @abc.abstractmethod
    def invert_ref_cdf(self, u: torch.Tensor) -> torch.Tensor:
        """Returns the inverse of the CDF of the unit reference 
        distribution evaluated at each value of u.
        
        Parameters
        ----------
        u:
            The values at which to evaluate the inverse CDF of the unit
            reference distribution.

        Returns
        -------
        :
            The inverse CDF of the reference distribution evaluated at
            each value of u.
        
        """
        return
    
    @abc.abstractmethod
    def log_joint_ref_pdf(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the log-PDF and gradient of the log-PDF of the 
        vector z, under the assumption the elements of z are independent
        and identically distributed according to the reference 
        distribution.

        Parameters
        ----------
        z:
            The vector at which to evaluate the log-PDF and gradient of 
            the log-PDF of the joint distribution.

        Returns
        -------
        :
            The value of the log-PDF and gradient of the log-PDF of z.
        
        """

    def map_to_ref(self, x: torch.Tensor) -> torch.Tensor:
        return (x-self.mu) / self.sigma

    def eval_cdf(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: might need to do some epsilon stuff in here.. 
        # (ask about this?)

        # Map x to the equivalent point on the unit reference distribution
        z = self.map_to_ref(x)
        u, f = self.eval_ref_cdf(self, z)

        # Account for any bounds on the cdf
        u = (u - self.left) / (self.right - self.left)
        
        # TODO: figure out what is going on here... (I don't get it)
        f = f / self.sigma / (self.right - self.left)
        
        return u, f

    def invert_cdf(self, u: torch.Tensor) -> torch.Tensor:

        u[torch.isinf(u)] = 1.0 - EPSILON
        u[torch.isnan(u)] = EPSILON

        # Map points into appropriate range
        u = self.left + u * (self.right-self.left)
        z = self.invert_ref_cdf(self, u)

        # Rescale points
        x = self.mu + self.sigma * z
        return x
        
    def eval_pdf(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        z = self.map_to_ref(x)
        f, g = self.eval_ref_pdf(z)
        
        # TODO: figure out what's going on here.
        f = f / self.sigma / (self.right - self.left)
        g = g / self.sigma**2 / (self.right - self.left)

        return f, g
        
    def log_joint_pdf(
        self, 
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        num_x = x.numel()

        z = self.map_to_ref(x)
        f, g = self.log_joint_ref_pdf(z)

        # TODO: figure out what is going on here.
        f = f - num_x * torch.log(self.sigma * (self.right - self.left))
        g = g / self.sigma
    
        return f, g
    
    def random(self, d: int, n: int) -> torch.Tensor:
        """Draws a set of samples from the reference distribution using
        the inverse CDF method.
        
        Parameters
        ----------
        d:
            The dimension of the samples.
        n:
            The number of samples to draw.

        Returns
        -------
        :
            The generated samples.

        """

        u = torch.rand(size=(n, d))
        u = self.left + (self.right - self.left) * u
        z = self.invert_cdf(self, u)
        return z
        
    def sobol(self, d: int, n: int) -> torch.Tensor:
        """Generates a set of QMC samples from the reference 
        distribution using a Sobol sequence.
        
        Parameters
        ----------
        d: 
            The dimension of the samples.
        n:
            The number of samples to generate.

        Returns
        :
            The generated samples.
        
        """

        S = torch.quasirandom.SobolEngine(dimension=d)
        u = S.draw(n)
        u = self.left + (self.right - self.left) * u
        z = self.invert_cdf(u)
        return z