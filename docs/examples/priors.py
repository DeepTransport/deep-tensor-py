import abc

import torch 
from torch import Tensor 


def compute_distance_matrix(X: Tensor, Y: Tensor) -> Tensor:
    """Returns a squared Euclidean distance matrix."""
    return (X[:, None, :] - Y[None, :, :]).square().sum(dim=2)


class Kernel(abc.ABC):
    
    @abc.abstractmethod
    def __call__(self, X: Tensor, Y: Tensor) -> Tensor:
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
        """Evaluates the covariance kernel at a set of points."""
        ds = compute_distance_matrix(X, Y)
        cov = self.std ** 2 * torch.exp(-ds/(2.0*self.lengthscale**2))
        return cov + 1e-8 * torch.eye(cov.shape[0])


class Prior(abc.ABC):

    @property 
    def mu(self) -> float | Tensor:
        return self._mu
    
    @mu.setter 
    def mu(self, val: float | Tensor) -> None:
        self._mu = val 
        return
    
    @property 
    def coef2node(self) -> Tensor:
        return self._coef2node
    
    @coef2node.setter 
    def coef2node(self, val: Tensor) -> None:
        self._coef2node = val 
        return
    
    @property 
    def dim(self) -> int:
        return self._dim
    
    @dim.setter 
    def dim(self, val: int) -> None:
        self._dim = val 
        return

    def transform(self, coefs: Tensor) -> Tensor:
        """Transforms a set of coefficient values to generate a 
        vector from the prior.
        """
        return self.mu + coefs @ self.coef2node.T
    
    def sample(self, n: int):
        """Returns a set of white noise samples."""
        return torch.randn((n, self.dim))


class GaussianRandomField(Prior):
    """A Gaussian random field with an arbitrary covariance kernel, in 
    arbitrary dimensions.
        
    Parameters
    ----------
    xs:
        A set of points at which the prior should be evaluated.
    mu: 
        Mean of field.
    kernel:
        The covariance kernel.
    
    """

    def __init__(
        self,
        xs: Tensor,
        mu: float | Tensor,
        kernel: Kernel,
        num_kl: int | None = None
    ):
        self.kl = num_kl is not None
        self.mu = mu
        self.xs = xs
        self.dim = num_kl if self.kl else xs.shape[0]  # type: ignore
        self.kernel = kernel
        self.coef2node = self._build_coef2node()
        return
    
    def _build_coef2node(self) -> Tensor:
        """Builds a matrix which, given the coefficients of the white 
        noise, returns the values of the field at each value of xs.
        """
        cov = self.kernel(self.xs, self.xs)
        if not self.kl:
            return torch.linalg.cholesky(cov)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        inds = torch.argsort(eigvals, descending=True)[:self.dim]
        coef2node = eigvecs[:, inds] * torch.sqrt(eigvals[inds])
        return coef2node#.flip(dims=(1,)) # TEMP!!


class ProcessConvolutionPrior(Prior):
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

    def __init__(
        self,
        xs: Tensor,
        ss: Tensor,
        mu: float = 0.0,
        r: float = 0.1
    ):
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
        d_sq = torch.sum((xxs-sss)**2, dim=2)
        coef2node = torch.exp(-(1.0 / (2.0 * self.r)) * d_sq)
        return coef2node