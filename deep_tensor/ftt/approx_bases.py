from typing import Tuple

import torch
from torch import Tensor

from ..domains import Domain
from ..polynomials import Basis1D


class ApproxBases():
    """Container of information on the approximation bases.
    
    This class contains information on the set of polynomial basis 
    functions used the construct the FTT, and the mapping from the 
    approximation domain to the domain of the polynomial basis.

    Parameters
    ----------
    polys:
        Tensor-product univariate polynomial basis functions, defined 
        on a local domain (generally (-1, 1)).
    domains:
        An invertible mapping between the approximation domain and the 
        domain of the polynomial basis functions.
    dim:
        The dimension of the domain.
    
    """

    def __init__(
        self, 
        polys: Basis1D | list[Basis1D],  # TODO: rename polys to bases?
        domain: Domain, 
        dim: int
    ):
        
        if isinstance(polys, Basis1D):
            polys = [polys]
        if len(polys) == 1:
            polys *= dim
        if len(polys) != dim:
            msg = ("Dimension of polynomials does not equal specified " 
                   + f"dimension (expected {dim}, got {len(polys)}).")
            raise Exception(msg)

        self.polys = polys
        self.domain = domain
        self.dim = dim
        return
    
    @staticmethod
    def check_indices_shape(indices: Tensor, xs: Tensor) -> None:
        """Confirms whether the length of a vector of indices is equal 
        to the dimension of a set of samples.
        """
        if indices.numel() != xs.shape[1]:
            msg = "Samples do not have the correct dimensions."
            raise Exception(msg)
        return

    def local2approx(
        self, 
        ls: Tensor, 
        indices: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Maps a set of samples drawn distributed in (a subset of) the 
        local domain to the approximation domain.
        
        Parameters
        ----------
        ls:
            An n * d matrix containing samples from the local domain.
        indices:
            The indices corresponding to the dimensions of the 
            local domain the samples live in (can be a subset of 
            {1, 2, ..., d}).

        Returns
        -------
        xs:
            An n * d matrix containing the corresponding samples in the 
            approximation domain.
        dxdls: 
            An n * d matrix containing the diagonal of the gradient of 
            the mapping from the local domain to the approximation 
            domain evaluated at each element of xs.

        """
        
        if indices is None:
            indices = torch.arange(self.dim)
        ApproxBases.check_indices_shape(indices, ls)

        xs = torch.empty_like(ls)
        dxdls = torch.empty_like(ls)
        for i, ls_i in enumerate(ls.T):
            xs[:, i], dxdls[:, i] = self.domain.local2approx(ls_i)

        return xs, dxdls

    def approx2local(
        self, 
        xs: Tensor, 
        indices: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Maps a set of samples from (a subset of) the approximation 
        domain to the local domain.
        
        Parameters
        ----------
        xs:
            An n * d matrix containing samples from the approximation 
            domain.
        indices:
            The indices corresponding to the dimensions of the 
            approximation domain the samples live in (can be a subset 
            of {1, 2, ..., d}).

        Returns
        -------
        ls:
            An n * d matrix containing the corresponding samples in the 
            local domain.
        dldxs: 
            An n * d matrix containing the diagonal of the gradient 
            of the mapping from the approximation domain to the 
            local domain evaluated at each element of ls.

        """
        
        if indices is None:
            indices = torch.arange(self.dim)
        ApproxBases.check_indices_shape(indices, xs)

        ls = torch.empty_like(xs)
        dldxs = torch.empty_like(xs)
        for i, xs_i in enumerate(xs.T):
            ls[:, i], dldxs[:, i] = self.domain.approx2local(xs_i)

        return ls, dldxs

    def local2approx_log_density(
        self,
        ls: Tensor,
        indices: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Computes the logarithm of the gradient, and derivative of 
        the gradient, of the transformation of a set of samples from 
        the local domain to the approximation domain.
        
        Parameters
        ----------
        ls:
            An n * d matrix containing samples from the local domain.

        Returns
        -------
        logdxdls:
            An n * d matrix containing the diagonal of the gradient of 
            the mapping from the local domain to the approximation 
            domain evaluated at each sample in ls.
        logdxdl2s:
            An n * d matrix containing the diagonal of the logarithm of 
            the derivative of the gradient of the mapping from the 
            local domain to the approximation domain evaluated at each 
            sample in ls.

        """
        
        if indices is None:
            indices = torch.arange(self.dim)
        ApproxBases.check_indices_shape(indices, ls)

        dlogxdls = torch.empty_like(ls)
        d2logxdl2s = torch.empty_like(ls)

        for i, ls_i in enumerate(ls.T):
            dlogxdl, d2logxdl2 = self.domain.local2approx_log_density(ls_i)
            dlogxdls[:, i], d2logxdl2s[:, i] = dlogxdl, d2logxdl2

        return dlogxdls, d2logxdl2s
    
    def approx2local_log_density(
        self,
        xs: Tensor,
        indices: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Computes the logarithm of the gradient, and derivative of 
        the gradient, of the transformation of a set of samples from 
        the approximation domain to the local domain.
        
        Parameters
        ----------
        xs:
            An n * d matrix containing samples from the approximation 
            domain.

        Returns
        -------
        logdldxs:
            An n * d matrix containing the diagonal of the gradient of 
            the mapping from the approximation domain to the local 
            domain evaluated at each sample in xs.
        logdxdl2s:
            An n * d matrix containing the diagonal of the logarithm of 
            the derivative of the gradient of the mapping from the 
            approximation domain to the local domain evaluated at each 
            sample in xs.
            
        """
        
        if indices is None:
            indices = torch.arange(self.dim)
        ApproxBases.check_indices_shape(indices, xs)

        logdldxs = torch.empty_like(xs)
        logd2ldx2s = torch.empty_like(xs)

        for i, xs_i in enumerate(xs.T):
            logdldx, logd2ldx2 = self.domain.approx2local_log_density(xs_i)
            logdldxs[:, i], logd2ldx2s[:, i] = logdldx, logd2ldx2

        return logdldxs, logd2ldx2s

    def sample_measure_local(self, n: int) -> Tuple[Tensor, Tensor]:
        """Generates a set of random variates from the local weighting 
        function.

        Parameters
        ----------
        n:
            Number of samples to generate.

        Returns
        -------
        ls:
            An n * d matrix containing samples drawn from the local
            weighting function.
        neglogwls:
            An n-dimensional vector containing the negative logarithm 
            of the weighting function evaluated at each sample.

        """ 

        ls = torch.zeros((n, self.dim))
        neglogwls = torch.zeros(n)
        
        for k in range(self.dim):
            ls[:, k] = self.polys[k].sample_measure(n)
            neglogwls -= self.polys[k].eval_log_measure(ls[:, k])
        
        return ls, neglogwls

    def eval_measure_potential_local(
        self, 
        ls: Tensor, 
        indices: Tensor | None = None
    ) -> Tensor:
        """Computes the negative logarithm of the weighting function 
        associated with (a subset of) the basis functions (defined in 
        the local domain).

        Parameters
        ----------
        ls:
            An n * d matrix containing samples from the local domain. 
        indices:
            The indices corresponding to the dimensions of the domain 
            the samples live in (can be a subset of {1, 2, ..., d}).

        Returns
        -------
        neglogwls: 
            An n-dimensional vector containg the negative logarithm of 
            the product of the weighting functions for each basis 
            evaluated at each input sample.

        """
            
        if indices is None:
            indices = torch.arange(self.dim)
        ApproxBases.check_indices_shape(indices, ls)

        neglogwls = torch.empty_like(ls)
        for i, ls_i in enumerate(ls.T):
            poly = self.polys[indices[i]]
            neglogwls[:, i] = -poly.eval_log_measure(ls_i)

        return neglogwls.sum(dim=1)

    def eval_measure_potential_local_grad(
        self, 
        ls: Tensor,
        indices: Tensor | None = None
    ):
        """Computes the gradient of the negative logarithm of the 
        weighting functions of (a subset of) the basis functions for a 
        given set of samples in the local domain.

        Parameters
        ----------
        ls: 
            An n * d matrix containing samples from the reference 
            distribution.
        indices:
            The indices corresponding to the dimensions of the 
            approximation domain the samples live in (can be a subset 
            of {1, 2, ..., d}). 
        
        Returns
        -------
        negloggradwls:
            An n * d matrix containing the negative logarithm of the 
            gradient of the weighting functions coresponding to each 
            sample in ls.
            
        """

        if indices is None:
            indices = torch.arange(self.dim)
        ApproxBases.check_indices_shape(indices, ls)
        
        negloggradwls = torch.empty_like(ls)
        for i, ls_i in enumerate(ls.T):
            poly = self.polys[indices[i]]
            negloggradwls[:, i] = -poly.eval_log_measure_deriv(ls_i)

        return negloggradwls

    def sample_measure(self, n: int) -> Tuple[Tensor, Tensor]:
        """Generates a set of samples from the approximation domain.
        
        Parameters
        ----------
        n:
            The number of samples to generate.
        
        Returns
        -------
        xs:
            An n * d matrix containing samples from the approximation 
            domain.
        neglogwxs:
            An n-dimensional vector containing the negative logarithm 
            of the weighting density (pushed forward into the 
            approximation domain) for each sample.
        
        """
        ls, neglogwls = self.sample_measure_local(n)
        xs, dxdls = self.local2approx(ls)
        neglogwxs = neglogwls + dxdls.log().sum(dim=1)
        return xs, neglogwxs
    
    def eval_measure_potential(
        self, 
        xs: Tensor, 
        indices: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Computes the target potential function and its gradient for 
        a set of samples from the approximation domain.
        
        Parameters
        ----------
        xs:
            An n * d matrix containing a set of samples from the 
            approximation domain.
        indices:
            The indices corresponding to the dimensions of the 
            approximation domain the samples live in (can be a subset 
            of {1, 2, ..., d}). 
        
        Returns
        -------
        neglogwxs:
            An n-dimensional vector containing the weighting function 
            evaluated at each element of xs.
        negloggradwxs:
            An n * d matrix containing the gradient of the negative 
            logarithm of each weighting function evaluated at each 
            element of xs.
        
        """
        
        if indices is None:
            indices = torch.arange(self.dim)
        ApproxBases.check_indices_shape(indices, xs)

        ls, dldxs = self.approx2local(xs, indices)
        
        neglogwls = self.eval_measure_potential_local(ls, indices)
        neglogwxs = neglogwls - dldxs.log().sum(dim=1)
        
        gradneglogwls = self.eval_measure_potential_local_grad(ls, indices)
        gradneglogwxs = gradneglogwls * dldxs
        return neglogwxs, gradneglogwxs
    
    def eval_measure(self, xs: Tensor) -> Tensor:
        """Computes the weighting function for a set of samples from 
        the approximation domain, with the domain mapping.

        Parameters
        ----------
        xs:
            An n * d matrix containing a set of n samples from the 
            approximation domain.
        
        Returns
        -------
        wxs:
            An n-dimensional vector containing the value of the 
            weighting function evaluated at each element in xs.

        """
        neglogwxs = self.eval_measure_potential(xs)[0]
        wxs = torch.exp(-neglogwxs)
        return wxs