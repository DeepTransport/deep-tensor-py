import abc

import torch 
from torch import Tensor 


def compute_distance_matrix(X: Tensor, Y: Tensor) -> Tensor:
    """Returns a squared Euclidean distance matrix."""
    return (X[:, None, :] - Y[None, :, :]).square().sum(dim=2)


class Kernel(abc.ABC):
    pass 


class SquaredExponential(Kernel):
    """The squared exponential (radial basis function) covariance 
    kernel.
    """

    def __init__(self, std: float, lengthscale: float):
        self.std = std 
        self.lengthscale = lengthscale
        return

    def __call__(self, X: Tensor, Y: Tensor) -> Tensor:
        """Evaluates the covariance kernel at a given set of points."""
        ds = compute_distance_matrix(X, Y)
        cov = self.std ** 2 * torch.exp(-ds/(2.0*self.lengthscale**2))
        return cov + 1e-8 * torch.eye(cov.shape[0])


class Prior(abc.ABC):

    def transform(self, coefs: Tensor) -> Tensor:
        """Transforms a set of coefficient values to generate a 
        vector from the prior.
        """
        return self.mu + self.coef2node @ coefs
    
    def sample(self, n: int):
        """Returns a set of white noise samples."""
        return torch.randn((n, self.dim))


class GaussianRandomField(Prior):

    def __init__(
        self,
        xs: Tensor,
        mu: float | Tensor,
        kernel: Kernel,
        num_kl: int | None = None
    ):
        """A two-dimensional process convolution prior.
        
        Parameters
        ----------
        xs:
            A set of points at which the prior should be evaluated.
        mu: 
            Mean of field.
        kernel:
            The covariance kernel.
        
        """
        if num_kl is None:
            num_kl = xs.shape[0]
        self.mu = mu
        self.xs = xs
        self.dim = num_kl
        self.kernel = kernel
        self.coef2node = self._build_coef2node()
        return
    
    def _build_coef2node(self):
        """Builds a matrix which, given the coefficients of the white 
        noise, returns the values of the field at each value of xs.
        """
        cov = self.kernel(self.xs, self.xs)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        inds = torch.argsort(eigvals, descending=True)[:self.dim]
        coef2node = eigvecs[:, inds] * torch.sqrt(eigvals[inds])
        return coef2node
    
    def transform(self, coefs: Tensor) -> Tensor:
        """Transforms a set of coefficient values to generate a 
        vector from the prior.
        """
        return self.mu + self.coef2node @ coefs


class ProcessConvolutionPrior(Prior):

    def __init__(
        self,
        xs: Tensor,
        ss: Tensor,
        mu: float = 0.0,
        r: float = 0.1
    ):
        """A two-dimensional process convolution prior.
        
        Parameters
        ----------
        xs:
            A set of points at which the prior should be evaluated.
        ss:
            The centres of the kernel functions.
        mu: 
            The mean.
        r:
            The radius of each kernel function.
        
        """
        self.xs = xs
        self.ss = ss
        self.mu = mu
        self.r = r
        self.dim = ss.shape[0]
        self.coef2node = self._build_coef2node()
        return
    
    def _build_coef2node(self):
        """Builds a matrix which, given the coefficients of the white 
        noise, returns the values of the field at each value of xs.
        """
        xxs = self.xs[:, None, :]
        sss = self.ss[None, :, :]
        d_sq = torch.sum((xxs-sss).square(), axis=2)
        coef2node = torch.exp(-(1.0 / (2.0 * self.r)) * d_sq)
        return coef2node
    
    def transform(self, coefs: Tensor) -> Tensor:
        """Transforms a set of coefficient values to generate a 
        vector from the prior.
        """
        return self.mu + self.coef2node @ coefs