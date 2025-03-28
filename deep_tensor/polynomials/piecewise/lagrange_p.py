from typing import Tuple
import warnings

import torch
from torch import Tensor

from .piecewise import Piecewise
from ..basis_1d import Basis1D
from ..spectral.jacobi_11 import Jacobi11
from ...constants import EPS
from ...tools import integrate


class LagrangeRef():
    
    def __init__(self, n: int):
        """Defines the reference Lagrange basis, in the reference
        domain [0, 1].

        Parameters
        ----------
        n: 
            The number of interpolation points to use.

        References
        ----------
        Berrut, J and Trefethen, LN (2004). Barycentric Lagrange 
        interpolation.

        """

        assert n > 2, "Value of n should be greater than 2."
        
        jacobi = Jacobi11(order=n-3)
        
        self.domain = torch.tensor([0.0, 1.0])
        self.cardinality = n
        self.es = torch.eye(n)
        self.nodes = torch.zeros(self.cardinality)
        self.nodes[1:-1] = 0.5 * (jacobi.nodes + 1.0)
        self.nodes[-1] = 1.0
        self.omega = self._compute_omegas()
        self.weights = self._compute_weights()
        self.mass = self._compute_mass()
        return
    
    def _compute_omegas(self) -> Tensor:
        """Computes the local Barycentric weights (see Berrut and 
        Trefethen, Eq. (3.2)).
        """
        omega = torch.zeros(self.cardinality)
        for i in range(self.cardinality):
            mask = torch.full((self.cardinality,), True)
            mask[i] = False
            omega[i] = torch.prod(self.nodes[i]-self.nodes[mask]) ** -1
        return omega
    
    def _compute_weights(self) -> Tensor:
        """Uses numerical integration to approximate the integral of 
        each basis function over the domain.
        """
        weights = torch.zeros(self.cardinality)
        for i in range(self.cardinality):
            f_i = lambda x: self.eval(self.es[i], x)
            weights[i] = integrate(f_i, *self.domain)
        return weights
    
    def _compute_mass(self) -> Tensor:
        """Uses numerical integration to approximate the mass matrix 
        (the integrals of the product of each pair of basis functions 
        over the domain).
        """
        mass = torch.zeros((self.cardinality, self.cardinality))
        for i in range(self.cardinality):
            for j in range(i, self.cardinality):
                e_i, e_j = self.es[i], self.es[j]
                f_ij = lambda ls: self.eval(e_i, ls) * self.eval(e_j, ls)
                integral = integrate(f_ij, *self.domain)
                mass[i, j], mass[j, i] = integral, integral
        return mass

    def eval(self, fls: Tensor, ls: Tensor) -> Tensor:
        """TODO: write docstring."""

        m = ls.numel()
        n = self.cardinality
        f = torch.zeros((m, ))

        inside = (ls >= self.domain[0]) & (ls <= self.domain[1])
        if not inside.all():
            msg = "Points outside of domain."
            warnings.warn(msg)

        n_in = inside.sum()

        diffs = ls[inside].repeat(n, 1).T - self.nodes.repeat(n_in, 1)
        diffs[diffs.abs() < EPS] = EPS  # avoid division by 0

        temp_m = self.omega.repeat(n_in, 1) / diffs

        # Evaluation of the internal interpolation
        f[inside] = (fls.repeat(n_in, 1) * temp_m).sum(dim=1) / temp_m.sum(dim=1)
        return f


class LagrangeP(Piecewise):
    r"""Higher-order piecewise Lagrange polynomials.

    Parameters
    ----------
    order:
        The degree of the polynomials.
    num_elems:
        The number of elements to use.

    Notes
    -----
    To construct a higher-order Lagrange basis, we divide the interval 
    $[0, 1]$ into `num_elems` equisized elements, and use a set of 
    Lagrange polynomials of degree $N=\,$`order` within each element.
     
    Within a given element, we choose a set of interpolation points, 
    $\{x_{n}\}_{n=0}^{N}$, which consist of the endpoints of the 
    element and the roots of the Jacobi polynomial of degree $N-3$ 
    (mapped into the domain of the element). Then, a given function can 
    be approximated (within the element) as
    $$
        f(x) \approx \sum_{n=0}^{N} f(x_{n})\ell_{n}(x),
    $$
    where the *Lagrange polynomials* $\{\ell_{n}(x)\}_{n=0}^{N}$ are 
    given by
    $$
        \ell_{n}(x) = \frac{\prod_{k = 0, k \neq n}^{N}(x-x_{k})}
            {\prod_{k = 0, k \neq n}^{N}(x_{n}-x_{k})}.
    $$
    To evaluate the interpolant, we use the second (true) form of the 
    Barycentric formula, which is more efficient and stable than the 
    above formula.

    We use piecewise Chebyshev polynomials of the second kind to 
    represent the corresponding (conditional) CDFs.

    References
    ----------
    Berrut, J and Trefethen, LN (2004). *[Barycentric Lagrange 
    interpolation](https://doi.org/10.1137/S0036144502417715).* 
    SIAM Review, **46**, 501--517.
    
    """

    def __init__(self, order: int, num_elems: int):

        if order == 1:
            msg = ("When 'order=1', Lagrange1 should be used " 
                   + "instead of LagrangeP.")
            raise Exception(msg)

        Piecewise.__init__(self, order, num_elems)
        
        self.local = LagrangeRef(self.order+1)

        n_nodes = self.num_elems * (self.local.cardinality - 1) + 1

        self.nodes = self._compute_nodes(n_nodes)

        unweighed_mass = torch.zeros((self.cardinality, self.cardinality))
        self.jac = self.elem_size / (self.local.domain[1] - self.local.domain[0])
        self.int_W = torch.zeros(self.cardinality)

        for i in range(self.num_elems):
            ind = (torch.arange(self.local.cardinality) 
                   + i * (self.local.cardinality-1))
            unweighed_mass[ind[:, None], ind[None, :]] += self.local.mass * self.jac
            self.int_W[ind] += self.local.weights * self.jac

        mass = unweighed_mass / self.domain_size
        self.mass_R = torch.linalg.cholesky(mass).T

        # map the function value y to each local element
        j = torch.arange(self.local.cardinality-1, n_nodes, self.local.cardinality-1)
        self.global2local = torch.vstack((
            torch.arange(self.cardinality-1).reshape(self.num_elems, self.local.cardinality-1).T, 
            j[None, :]
        )).T

        return
    
    @property
    def int_W(self) -> Tensor:
        return self._int_W
    
    @int_W.setter
    def int_W(self, value: Tensor) -> None:
        self._int_W = value 
        return
    
    @property 
    def nodes(self) -> Tensor:
        return self._nodes 

    @nodes.setter 
    def nodes(self, value: Tensor) -> None:
        self._nodes = value 
        return
    
    @property
    def cardinality(self):
        return self.nodes.numel()
    
    @property
    def domain(self):
        return torch.tensor([-1.0, 1.0])
    
    @property 
    def mass_R(self) -> Tensor:
        return self._mass_R
    
    @mass_R.setter
    def mass_R(self, value: Tensor) -> None:
        self._mass_R = value 
        return
    
    @staticmethod
    def _adjust_dls(dls: Tensor) -> Tensor:
        """Ensures that no values of the dls matrix are equal to 0."""
        dls[(dls >= 0) & (dls.abs() < EPS)] = EPS
        dls[(dls < 0) & (dls.abs() < EPS)] = -EPS 
        return dls
    
    def _compute_nodes(self, n_nodes: int) -> Tensor:
        """Computes the values of the global nodes. 
        TODO: give more detail on this.
        """
        nodes = torch.zeros(n_nodes)
        for i in range(self.num_elems):
            ind = (torch.arange(self.local.cardinality) 
                   + i * (self.local.cardinality-1))
            nodes[ind] = self.grid[i] + self.elem_size * self.local.nodes
        return nodes

    @Basis1D._check_samples
    def eval_basis(self, ls: Tensor) -> Tensor:
        
        n_ls = ls.numel()
        ps = torch.zeros((n_ls, self.cardinality))
        
        left_inds = self.get_left_hand_inds(ls)
        ls_local = self.map_to_element(ls, left_inds)
        
        dls = ls_local[:, None] - self.local.nodes
        dls = self._adjust_dls(dls)
        sum_terms = self.local.omega / dls
        ps_loc = sum_terms / sum_terms.sum(1, keepdim=True)

        ii = torch.arange(n_ls).repeat_interleave(self.local.cardinality)
        jj = self.global2local[left_inds].flatten()
        ps[ii, jj] = ps_loc.flatten()
        return ps
    
    @Basis1D._check_samples
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:
        
        self._check_in_domain(ls)

        n_ls = ls.numel()
        dpdls = torch.zeros((n_ls, self.cardinality))
        
        left_inds = self.get_left_hand_inds(ls)
        ls_local = self.map_to_element(ls, left_inds)
        
        dls = ls_local[:, None] - self.local.nodes
        dls = self._adjust_dls(dls)
        
        sum_terms = self.local.omega / dls
        sum_terms_sq = self.local.omega / dls.square()

        coefs_b = 1.0 / torch.sum(sum_terms, dim=1, keepdim=True)
        coefs_a = torch.sum(sum_terms_sq, dim=1, keepdim=True) * coefs_b.square()

        dpdls_loc = (coefs_a * sum_terms - coefs_b * sum_terms_sq) / self.jac
        ii = torch.arange(n_ls).repeat_interleave(self.local.cardinality)
        jj = self.global2local[left_inds].flatten()
        dpdls[ii, jj] = dpdls_loc.flatten()
        return dpdls