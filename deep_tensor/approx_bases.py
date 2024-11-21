from typing import Tuple
import warnings

import torch

from .domains import BoundedDomain
from .polynomials import Basis1D


class ApproxBases():
    """This class contains tensor-product polynomial basis functions 
    and mappings from the approximation domain to the reference domain.

    Properties
    ----------
    dim:
        The dimension of the approximation domain.
    polys:
        Tensor-product univariate polynomial basis functions, defined 
        on the reference domain.
    domains:
        The approximation domain. Contains methods for mapping to the 
        reference domain.
    
    """

    # TODO: check whether the type annotations are correct.
    def __init__(
        self, 
        polys: list[Basis1D], 
        in_domains: list[BoundedDomain], 
        dim: int
    ):

        self.num_polys = len(polys) 
        self.num_domains = len(in_domains)
        self.dim = dim

        if self.num_polys > 1 and self.num_domains > 1:
            if self.num_polys != self.num_domains:
                msg = "Dimension of domains does not equal dimension of bases."
                raise Exception(msg)
            if self.num_polys != self.dim:
                msg = "Dimension does not match number of polynomial bases."
                warnings.warn(msg)
                self.dim = self.num_polys  # now everything is the same.

        else:
            max_d = max(self.num_polys, self.num_domains)
            if max_d > 1 and max_d != self.dim:
                msg = "Dimension does not match the number of polynomial bases."
                warnings.warn(msg)
                self.dim = max_d

            # TODO: figure out whether the below code needs to copy 
            # the bases (could be an issue)
            if self.num_domains == 1:
                in_domains *= self.dim
            if self.num_polys == 1:
                polys *= self.dim

        # oned_domains in TC's code
        self.in_domains = in_domains 
        # oneds in TC's code
        self.polys = polys

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
        :
            The cardinalities of each polynomial.
        
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
            del self.in_domains[i]
        return

    def reference2domain(
        self, 
        rs: torch.Tensor, 
        indices: torch.Tensor|None=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Maps a set of samples drawn from (a subset of) the reference 
        measure to the approximation domain.
        
        Parameters
        ----------
        rs:
            An n * d matrix containing samples from the reference 
            domain.
        indices:
            The indices corresponding to the dimensions of the 
            reference domain the samples live in (can be a subset of 
            {1, 2, ..., d}).

        Returns
        -------
        xs
            An n * d matrix containing the corresponding samples in the 
            approximation domain.
        dxdrs: 
            An n * d matrix containing the diagonal of the gradient of 
            the mapping from the reference domain to the approximation 
            domain evaluated at each sample.

        """
        
        if indices is None:
            indices = torch.arange(self.dim)
        
        if len(indices) != rs.shape[1]:
            msg = "Reference samples do not have the correct dimension."
            raise Exception(msg)

        xs = torch.empty_like(rs)
        dxdrs = torch.empty_like(rs)
        for i, r in enumerate(rs.T):
            domain = self.in_domains[indices[i]]
            xs[:, i], dxdrs[:, i] = domain.reference2domain(r)

        return xs, dxdrs

    def domain2reference(
        self, 
        xs: torch.Tensor, 
        indices: torch.Tensor|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps a set of samples from (a subset of) the approximation 
        domain to the reference domain.
        
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
        rs:
            An n * d matrix containg the corresponding samples from the 
            reference domain.
        drdxs: 
            An n * d matrix containing the (diagonal of the) gradient 
            of the mapping from the approximation domain to the 
            reference domain evaluated at each sample.

        """
        
        if indices is None:
            indices = torch.arange(self.dim)
        
        if len(indices) != xs.shape[1]:
            msg = "xs does not have the correct dimensions."
            raise Exception(msg)

        rs = torch.empty_like(xs)
        drdxs = torch.empty_like(xs)
        for i, x in enumerate(xs.T):
            domain = self.in_domains[indices[i]]
            rs[:, i], drdxs[:, i] = domain.domain2reference(x)

        return rs, drdxs

    def reference2domain_log_density(
        self,
        rs: torch.Tensor,
        indices: torch.Tensor|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Not entirely sure what this is doing yet...
        
        logarithm of INDUCED density of mapping and its gradient. 
        This could be something to do with pushforward stuff?, jacobian etc"""
        
        if indices is None:
            indices = torch.arange(self.dim)

        if len(indices) != rs.shape[1]:
            msg = "zs does not have the correct dimensions."
            raise Exception(msg)

        dlogxdrs = torch.empty_like(rs)
        d2logxdr2s = torch.empty_like(rs)

        for i, r in enumerate(rs.T):
            domain = self.in_domains[indices[i]]
            dlogxdrs[:, i], d2logxdr2s[:, i] = domain.reference2domain_log_density(r)

        return dlogxdrs, d2logxdr2s
    
    def domain2reference_log_density(
        self,
        xs: torch.Tensor,
        indices: torch.Tensor|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Not entirely sure what this is doing yet..."""
        
        if indices is None:
            indices = torch.arange(self.dim)

        if len(indices) != xs.shape[1]:
            msg = "zs does not have the correct dimensions."
            raise Exception(msg)

        dlogrdxs = torch.empty_like(xs)
        d2logrdx2s = torch.empty_like(xs)

        for i, x in enumerate(xs.T):
            domain = self.in_domains[indices[i]]
            dlogrdxs[:, i], d2logrdx2s[:, i] = domain.domain2reference_log_density(x)

        return dlogrdxs, d2logrdx2s

    def sample_measure_reference(
        self, 
        n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a set of random variates from the weighting 
        function of the (product-form) reference distribution.

        Parameters
        ----------
        n:
            Number of samples to generate.

        Returns
        -------
        rs:
            An n * d matrix containing samples drawn from the reference
            distribution.
        neglogfs:
            An n-dimensional vector containing the negative logarithm 
            of the weighting function evaluated at each sample.

        """ 

        rs = torch.zeros((n, self.dim))
        neglogfs = torch.zeros(n)
        
        for k in range(self.dim):
            rs[:, k] = self.polys[k].sample_measure(n)
            neglogfs -= self.polys[k].eval_log_measure(rs[:, k])
        
        return rs, neglogfs

    def eval_measure_potential_reference(
        self, 
        rs: torch.Tensor, 
        indices: torch.Tensor|None=None
    ) -> torch.Tensor:
        """Computes the negative logarithm of the weighting function 
        associated with (a subset of) the basis functions for a set of 
        reference variables.

        Parameters
        ----------
        rs:
            An n * d matrix containing samples from the reference 
            distribution.
        indices:
            The indices corresponding to the dimensions of the 
            approximation domain the samples live in (can be a subset 
            of {1, 2, ..., d}).

        Returns
        -------
        neglogref: 
            An n-dimensional vector containg the negative logarithm of 
            the product of the weighting functions for each basis 
            evaluated at each input sample.

        """
            
        if indices is None:
            indices = torch.arange(self.dim)

        if len(indices) != rs.shape[1]:
            msg = "zs does not have the correct dimension."
            raise Exception(msg)

        ws = torch.empty_like(rs)
        for i, r in enumerate(rs.T):
            ws[:, i] = -self.polys[i].eval_log_measure(r)

        # Radon-Nikodym derivative w.r.t. the base measure 
        # (TODO: figure out why)
        neglogref = torch.sum(ws, dim=1)
        return neglogref

    def eval_measure_potential_reference_grad(
        self, 
        rs: torch.Tensor,
        indices: torch.Tensor|None=None
    ):
        """Computes the gradient of the negative logarithm of the 
        weighting functions of (a subset of) the basis functions for a 
        given set of samples from the reference distribution.

        Parameters
        ----------
        rs: 
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
            functions coresponding to each sample in zs.

        """

        if indices is None:
            indices = torch.arange(self.dim)

        if len(indices) != rs.shape[1]:
            msg = "zs does not have the correct dimension."
            raise Exception(msg)
        
        gs = torch.empty_like(rs)

        for i, z in enumerate(rs.T):
            gs[:, i] = -self.polys[indices[i]].eval_log_measure_deriv(z)

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
        neglogfx:   
            An n-dimensional vector containing the negative logarithm 
            of the pushforward density of each sample.
        
        """

        rs, neglogfrs = self.sample_measure_reference(n)
        xs, dxdrs = self.reference2domain(rs)

        neglogfxs = neglogfrs + torch.sum(torch.log(dxdrs), dim=1)
        return xs, neglogfxs
    
    def eval_measure(self, xs: torch.Tensor) -> torch.Tensor:
        """Computes the density of the reference measure for a set of 
        variates from the input domain, with the domain mapping.

        TODO: finish docstring.
        """

        fx, _ = self.eval_measure_potential(xs)
        fx = torch.exp(-fx)
        return fx
    
    def eval_measure_potential(
        self, 
        xs: torch.Tensor,
        indices: torch.Tensor|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the negative logarithm of the reference measure and
        its gradient for variables in the input domain, with the domain 
        mapping.
        
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
        TODO
        """
        
        if indices is None:
            indices = torch.arange(self.dim)
        
        if len(indices) != xs.shape[1]:
            msg = "xs does not have the correct dimension."
            raise Exception(msg)

        zs, dzdx = self.domain2reference(xs, indices)
        
        fz = self.eval_measure_potential_reference(zs, indices)
        fx = fz + torch.sum(torch.log(dzdx), 1)  # TODO: check if this works.
        
        gz = self.eval_measure_potential_reference_grad(zs, indices)
        raise NotImplementedError("TODO: figure out what is going on in here.")
        gx = gz + -1

        return fx, gx