from typing import Tuple

import torch

from .domains import Domain
from .polynomials import Basis1D


PolyType = Basis1D | list[Basis1D]
DomainType = Domain | list[Domain]


class ApproxBases():

    def __init__(self, polys: PolyType, domains: DomainType, dim: int):
        """Class containing a set of tensor-product polynomial basis 
        functions and mappings between the local domain and 
        approximation domain.

        Parameters
        ----------
        polys:
            Tensor-product univariate polynomial basis functions, 
            defined on a local domain (generally [-1, 1]).
        domains:
            The approximation domain in each dimension.
        dim:
            The dimension of the approximation domain.
        
        """
        
        if isinstance(polys, Basis1D):
            polys = [polys]
        if isinstance(domains, Domain):
            domains = [domains]

        if len(domains) == 1:
            domains *= dim
        if len(polys) == 1:
            polys *= dim

        if len(domains) != dim:
            msg = ("Dimension of domain does not equal specified " 
                   + f"dimension (expected {dim}, got {len(domains)}).")
            raise Exception(msg)
        
        if len(polys) != dim:
            msg = ("Dimension of polynomials does not equal specified " 
                   + f"dimension (expected {dim}, got {len(polys)}).")
            raise Exception(msg)

        self.dim = dim
        self.domains = domains 
        self.polys = polys

        return

    def get_cardinalities(
        self, 
        indices: torch.Tensor|None=None
    ) -> torch.Tensor:
        """Returns the cardinalities of (a subset of) the polynomials 
        that form the current basis.

        Parameters
        ----------
        indices:
            The indices of the polynomials whose cardinalities should 
            be returned. If this is not passed in, the cardinalities 
            of all polynomials of the basis are returned.

        Returns
        -------
        cardinalities:
            A d-dimensional vector containing the cardinalities of each 
            polynomial.
        
        """

        if indices is None:
            indices = torch.arange(self.dim)
        
        return torch.tensor([self.polys[i].cardinality for i in indices])
        
    def duplicate_bases(self):
        raise NotImplementedError()
    
    def remove_bases(self, indices: torch.Tensor) -> None:
        """Removes a set of bases.
        
        Parameters
        ----------
        indices:
            The indices of the bases to remove.
        
        """
        for i in indices:
            del self.polys[i]
            del self.domains[i]
        return

    def local2approx(
        self, 
        ls: torch.Tensor, 
        indices: torch.Tensor|None=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        
        if len(indices) != ls.shape[1]:
            msg = "Samples do not have the correct dimension."
            raise Exception(msg)

        xs = torch.empty_like(ls)
        dxdls = torch.empty_like(ls)
        for i, ls_i in enumerate(ls.T):
            domain = self.domains[indices[i]]
            xs[:, i], dxdls[:, i] = domain.local2approx(ls_i)

        return xs, dxdls

    def approx2local(
        self, 
        xs: torch.Tensor, 
        indices: torch.Tensor|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            An n * d matrix containg the corresponding samples in the 
            local domain.
        dldxs: 
            An n * d matrix containing the (diagonal of the) gradient 
            of the mapping from the approximation domain to the 
            local domain evaluated at each element of ls.

        """
        
        if indices is None:
            indices = torch.arange(self.dim)
        
        if len(indices) != xs.shape[1]:
            msg = "Samples do not have the correct dimensions."
            raise Exception(msg)

        ls = torch.empty_like(xs)
        dldxs = torch.empty_like(xs)
        for i, xs_i in enumerate(xs.T):
            domain = self.domains[indices[i]]
            ls[:, i], dldxs[:, i] = domain.approx2local(xs_i)

        return ls, dldxs

    def local2approx_log_density(
        self,
        ls: torch.Tensor,
        indices: torch.Tensor|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the logarithm of the gradient, and derivative of 
        the gradient, of the transformation of a set of samples from 
        the local domain to the approximation domain.
        
        Parameters
        ----------
        ls:
            An n * d matrix containing samples from the local domain.

        Returns
        -------
        dlogxdls:
            An n-dimensional vector containing...?
        """
        
        if indices is None:
            indices = torch.arange(self.dim)

        if len(indices) != ls.shape[1]:
            msg = "Samples do not have the correct dimensions."
            raise Exception(msg)

        dlogxdls = torch.empty_like(ls)
        d2logxdl2s = torch.empty_like(ls)

        for i, ls_i in enumerate(ls.T):
            domain = self.domains[indices[i]]
            dlogxdl, d2logxdl2 = domain.local2approx_log_density(ls_i)
            dlogxdls[:, i], d2logxdl2s[:, i] = dlogxdl, d2logxdl2

        return dlogxdls, d2logxdl2s
    
    def approx2local_log_density(
        self,
        xs: torch.Tensor,
        indices: torch.Tensor|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: write docstring."""
        
        if indices is None:
            indices = torch.arange(self.dim)

        if len(indices) != xs.shape[1]:
            msg = "Samples do not have the correct dimensions."
            raise Exception(msg)

        dlogldxs = torch.empty_like(xs)
        d2logldx2s = torch.empty_like(xs)

        for i, xs_i in enumerate(xs.T):
            domain = self.domains[indices[i]]
            dlogldx, d2logldx2 = domain.approx2local_log_density(xs_i)
            dlogldxs[:, i], d2logldx2s[:, i] = dlogldx, d2logldx2

        return dlogldxs, d2logldx2s

    def sample_measure_local(
        self, 
        n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        ls: torch.Tensor, 
        indices: torch.Tensor|None=None
    ) -> torch.Tensor:
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

        if len(indices) != ls.shape[1]:
            msg = "Samples do not have the correct dimension."
            raise Exception(msg)

        neglogwls = torch.empty_like(ls)
        for i, ls_i in enumerate(ls.T):
            neglogwls[:, i] = -self.polys[i].eval_log_measure(ls_i)

        return neglogwls.sum(dim=1)

    def eval_measure_potential_local_grad(
        self, 
        ls: torch.Tensor,
        indices: torch.Tensor|None=None
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
        gs:
            The gradient of the negative logarithm of the weighting 
            functions coresponding to each sample in ls.

        TODO: tidy up the returns of this docstring.
            
        """

        if indices is None:
            indices = torch.arange(self.dim)

        if len(indices) != ls.shape[1]:
            msg = "Samples do not have the correct dimension."
            raise Exception(msg)
        
        gs = torch.empty_like(ls)
        for i, ls_i in enumerate(ls.T):
            poly = self.polys[indices[i]]
            gs[:, i] = -poly.eval_log_measure_deriv(ls_i)

        return gs

    def sample_measure(
        self, 
        n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        xs: torch.Tensor,
        indices: torch.Tensor|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            The weighting function evaluated at each element of xs.
        gxs:
            TODO: change the naming etc here.

        TODO: check this with TC (MATLAB version is different)

        """
        
        if indices is None:
            indices = torch.arange(self.dim)
        
        if len(indices) != xs.shape[1]:
            msg = "xs does not have the correct dimension."
            raise Exception(msg)

        ls, dldxs = self.approx2local(xs, indices)
        
        neglogwls = self.eval_measure_potential_local(ls, indices)
        neglogwxs = neglogwls - dldxs.log().sum(dim=1)  # TODO: check this
        
        grs = self.eval_measure_potential_local_grad(ls, indices)
        gxs = grs * dldxs

        return neglogwxs, gxs
    
    def eval_measure(self, xs: torch.Tensor) -> torch.Tensor:
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