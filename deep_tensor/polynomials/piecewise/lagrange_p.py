from typing import Tuple
import warnings

import torch
from torch import Tensor

from .piecewise import Piecewise
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

    def __init__(self, order, num_elems):

        if order == 1:
            msg = ("When 'order=1', Lagrange1 should be used " 
                   + "instead of LagrangeP.")
            raise Exception(msg)

        Piecewise.__init__(self, order, num_elems)
        
        self.local = LagrangeRef(self.order+1)

        n_nodes = self.num_elems * (self.local.cardinality - 1) + 1

        self._nodes = self._compute_nodes(n_nodes)

        unweighed_mass = torch.zeros((self.cardinality, self.cardinality))
        self.jac = self.elem_size / (self.local.domain[1] - self.local.domain[0])
        self._int_W = torch.zeros(self.cardinality)

        for i in range(self.num_elems):
            ind = (torch.arange(self.local.cardinality) 
                   + i * (self.local.cardinality-1))
            unweighed_mass[ind[:, None], ind[None, :]] += self.local.mass * self.jac
            self._int_W[ind] += self.local.weights * self.jac

        mass = unweighed_mass / self.domain_size
        self._mass_R = torch.linalg.cholesky(mass).T

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
    def mass_R(self) -> Tensor:
        return self._mass_R
    
    @mass_R.setter
    def mass_R(self, value: Tensor) -> None:
        self._mass_R = value 
        return
    
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

    def eval_basis(self, ls: Tensor) -> Tensor:
        
        basis_vals = torch.zeros((ls.numel(), self.cardinality))

        if not torch.all(inside := self.in_domain(ls)):
            warnings.warn("Some points are outside the domain.")

        if not torch.any(inside):
            return basis_vals
        
        left_inds = self.get_left_hand_inds(ls[inside])

        ls_local = (ls[inside] - self.grid[left_inds]) / self.elem_size
        
        diffs = ls_local.repeat(self.local.cardinality, 1).T - self.local.nodes.repeat(inside.sum(), 1)
        diffs[diffs.abs() < EPS] = EPS

        temp_m = self.local.omega.repeat(inside.sum(), 1) / diffs
        lbs = temp_m / temp_m.sum(1, keepdim=True)
        coi = self.global2local[left_inds].T.flatten()
        roi = inside.nonzero().flatten().repeat(self.local.cardinality)
        
        # Evaluation of the internal interpolation
        basis_vals[roi, coi] = lbs.T.flatten()
        return basis_vals
    
    def eval_basis_deriv(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        
        deriv_vals = torch.zeros((ls.numel(), self.cardinality))

        if not torch.all(inside := self.in_domain(ls)):
            warnings.warn("Some points are outside the domain.")

        if not torch.any(inside):
            return deriv_vals
        
        left_inds = self.get_left_hand_inds(ls[inside])

        ls_local = (ls[inside] - self.grid[left_inds]) / self.elem_size
        
        diffs = ls_local.repeat(self.local.cardinality, 1).T - self.local.nodes.repeat(inside.sum(), 1)
        diffs[diffs.abs() < EPS] = EPS

        temp_m1 = self.local.omega.repeat(inside.sum(), 1) / diffs
        temp_m2 = self.local.omega.repeat(inside.sum(), 1) / (diffs ** 2)

        a = 1.0 / torch.sum(temp_m1, dim=1, keepdim=True)
        b = torch.sum(temp_m2, dim=1, keepdim=True) * torch.pow(a, 2)

        lbs = (temp_m1 * b - temp_m2 * a) / self.jac
        coi = self.global2local[left_inds].T.flatten()
        roi = inside.nonzero().flatten().repeat(self.local.cardinality)
        
        # Evaluation of the internal interpolation
        deriv_vals[roi, coi] = lbs.T.flatten()
        return deriv_vals