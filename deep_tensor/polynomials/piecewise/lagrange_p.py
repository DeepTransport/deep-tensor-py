import torch
from torch import Tensor

from .piecewise import Piecewise
from ..spectral.jacobi_11 import Jacobi11
from ...constants import EPS
from ...tools import integrate


class _LagrangeRef():
    
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
        self.domain_size = self.domain[1] - self.domain[0]
        self.cardinality = n
        self.es = torch.eye(n)
        self.nodes = torch.zeros(self.cardinality)
        self.nodes[1:-1] = 0.5 * (jacobi.nodes + 1.0)
        self.nodes[-1] = 1.0
        self._compute_omegas()
        self._compute_weights()
        self._compute_mass()
        return
    
    def _compute_omegas(self) -> None:
        """Computes the local Barycentric weights (see Berrut and 
        Trefethen, Eq. (3.2)).
        """
        self.omega = torch.zeros(self.cardinality)
        for i in range(self.cardinality):
            mask = torch.full((self.cardinality,), True)
            mask[i] = False
            self.omega[i] = torch.prod(self.nodes[i]-self.nodes[mask]) ** -1
        return
    
    def _compute_weights(self) -> None:
        """Uses numerical integration to approximate the integral of 
        each basis function over the domain.
        """
        self.weights = torch.zeros(self.cardinality)
        for i in range(self.cardinality):
            f_i = lambda x: self._eval(self.es[i], x)
            self.weights[i] = integrate(f_i, *self.domain)
        return
    
    def _compute_mass(self) -> None:
        """Uses numerical integration to approximate the mass matrix 
        (the integrals of the product of each pair of basis functions 
        over the domain).
        """
        self.mass = torch.zeros((self.cardinality, self.cardinality))
        for i in range(self.cardinality):
            for j in range(i, self.cardinality):
                e_i, e_j = self.es[i], self.es[j]
                f_ij = lambda ls: self._eval(e_i, ls) * self._eval(e_j, ls)
                integral = integrate(f_ij, *self.domain)
                self.mass[i, j] = self.mass[j, i] = integral
        return

    def _eval(self, coefs: Tensor, ls: Tensor) -> Tensor:
        """Returns the value of the polynomial basis at each of a set 
        of points.
        
        Parameters
        ----------
        coefs:
            An m-dimensional vector containing the coefficient 
            associated with each Lagrange polynomial.
        ls:
            An n-dimensional vector containing a set of points at which 
            to evaluate the polynomial basis.
        
        Returns
        -------
        ps: 
            An n-dimensional vector containing the value of the basis 
            evaluated at each point in ls.
        
        """
        dls = ls[:, None] - self.nodes
        dls = LagrangeP._adjust_dls(dls)
        sum_terms = self.omega / dls
        ps = (coefs * sum_terms).sum(dim=1) / sum_terms.sum(dim=1)
        return ps


class LagrangeP(Piecewise):
    r"""Higher-order piecewise Lagrange polynomials.

    Parameters
    ----------
    order:
        The degree of the polynomials, $n$.
    num_elems:
        The number of elements to use.

    Notes
    -----
    To construct a higher-order Lagrange basis, we divide the interval 
    $[0, 1]$ into `num_elems` equisized elements, and use a set of 
    Lagrange polynomials of degree $n=\,$`order` within each element.
     
    Within a given element, we choose a set of interpolation points, 
    $\{x_{j}\}_{j=0}^{n}$, which consist of the endpoints of the 
    element and the roots of the Jacobi polynomial of degree $n-3$ 
    (mapped into the domain of the element). Then, a given function can 
    be approximated (within the element) as
    $$
        f(x) \approx \sum_{j=0}^{n} f(x_{j})p_{j}(x),
    $$
    where the *Lagrange polynomials* $\{p_{j}(x)\}_{j=0}^{n}$ are 
    given by
    $$
        p_{j}(x) = \frac{\prod_{k = 0, k \neq j}^{n}(x-x_{k})}
            {\prod_{k = 0, k \neq j}^{n}(x_{j}-x_{k})}.
    $$
    To evaluate the interpolant, we use the second (true) form of the 
    Barycentric formula, which is more efficient and stable than the 
    above formula.

    We use piecewise Chebyshev polynomials of the second kind to 
    represent the (conditional) CDFs corresponding to the higher-order 
    Lagrange representation of (the square root of) the target density 
    function.

    References
    ----------
    Berrut, J and Trefethen, LN (2004). *[Barycentric Lagrange 
    interpolation](https://doi.org/10.1137/S0036144502417715).* 
    SIAM Review **46**, 501--517.
    
    """

    def __init__(self, order: int, num_elems: int):

        if order == 1:
            msg = ("When 'order=1', Lagrange1 should be used " 
                   + "instead of LagrangeP.")
            raise Exception(msg)

        Piecewise.__init__(self, order, num_elems)
        self.local = _LagrangeRef(self.order+1)

        # Define Jacobian of mapping from the domain of the LagrangeRef 
        # polynomial to an element
        self.jac = self.elem_size / self.local.domain_size

        self._compute_nodes()
        self._compute_mass()
        self._compute_int_W()

        # elem_nodes[i] returns the nodes corresponding to element i
        self.elem_nodes = torch.tensor([
            range(n*self.order, (n+1)*self.order+1) 
            for n in range(self.num_elems)])

        return
    
    @property
    def int_W(self) -> Tensor:
        return self._int_W
    
    @int_W.setter
    def int_W(self, value: Tensor) -> None:
        self._int_W = value 
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
    
    def _compute_nodes(self) -> None:
        """Computes the values of the global nodes. The grid of the 
        polynomial is divided into 'num_elems' equispaced elements. 
        Within each element, the nodes of the Jacobi polynomial of the 
        appropriate order are used.
        """
        n_loc = self.local.cardinality
        n_nodes = self.num_elems * (n_loc-1) + 1
        nodes = torch.zeros(n_nodes)
        for i in range(self.num_elems):
            inds_elem = torch.arange(n_loc) + i * (n_loc-1)
            nodes[inds_elem] = self.grid[i] + self.elem_size * self.local.nodes    
        self.nodes = nodes
        return
    
    def _compute_mass(self) -> None:
        """Computes the mass matrix and its Cholesky factor."""
        n_loc = self.local.cardinality
        mass_elem = self.local.mass * (0.5 * self.jac)
        self.mass = torch.zeros((self.cardinality, self.cardinality))
        for i in range(self.num_elems):
            inds_elem = torch.arange(n_loc) + i * (n_loc-1)
            self.mass[inds_elem[:, None], inds_elem[None, :]] += mass_elem
        self.mass_R = torch.linalg.cholesky(self.mass).T
        return
    
    def _compute_int_W(self) -> None:
        """Computes the integration operator."""
        n_loc = self.local.cardinality
        weights_elem = self.local.weights * (0.5 * self.jac)
        self.int_W = torch.zeros(self.cardinality)
        for i in range(self.num_elems):
            inds_elem = torch.arange(n_loc) + i * (n_loc-1)
            self.int_W[inds_elem] += weights_elem
        return

    def eval_basis(self, ls: Tensor) -> Tensor:
        
        self._check_in_domain(ls)
        
        n_ls = ls.numel()
        ps = torch.zeros((n_ls, self.cardinality))
        
        left_inds = self.get_left_hand_inds(ls)
        ls_local = self.map_to_element(ls, left_inds)
        
        dls = ls_local[:, None] - self.local.nodes
        dls = self._adjust_dls(dls)
        sum_terms = self.local.omega / dls
        ps_loc = sum_terms / sum_terms.sum(1, keepdim=True)

        ii = torch.arange(n_ls).repeat_interleave(self.local.cardinality)
        jj = self.elem_nodes[left_inds].flatten()
        ps[ii, jj] = ps_loc.flatten()
        return ps
    
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
        jj = self.elem_nodes[left_inds].flatten()
        dpdls[ii, jj] = dpdls_loc.flatten()
        return dpdls