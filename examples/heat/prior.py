import torch 
from torch import Tensor 


class ProcessConvolutionPrior(object):

    def __init__(
        self,
        xs: Tensor,
        ss: Tensor,
        mu: float = -5,
        r: float = 1/16
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