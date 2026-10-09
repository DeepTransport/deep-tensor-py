import torch
from torch import Tensor

from .piecewise import Piecewise
from ..spectral.jacobi_11 import Jacobi11
from ...constants import EPS
from ...integration import integrate


class _LagrangeRef():
    
    def __init__(self, n: int, device: torch.device):
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
        
        self._device = device
        self._domain = torch.tensor([0.0, 1.0], device=self._device)
        self._domain_size = self._domain[1] - self._domain[0]
        self._cardinality = n
        self._es = torch.eye(n, device=self._device)
        self._nodes = torch.zeros(self._cardinality, device=self._device)
        self._nodes[1:-1] = 0.5 * (jacobi._nodes + 1.0)
        self._nodes[-1] = 1.0
        self._compute_omegas()
        self._compute_weights()
        self._compute_mass()
        return
    
    def _compute_omegas(self) -> None:
        """Computes the local Barycentric weights (see Berrut and 
        Trefethen, Eq. (3.2)).
        """
        self._omega = torch.zeros(self._cardinality, device=self._device)
        for i in range(self._cardinality):
            mask = torch.full((self._cardinality,), True, device=self._device)
            mask[i] = False
            self._omega[i] = torch.prod(self._nodes[i]-self._nodes[mask]) ** -1
        return
    
    def _compute_weights(self) -> None:
        """Uses numerical integration to approximate the integral of 
        each basis function over the domain.
        """
        self._weights = torch.zeros(self._cardinality, device=self._device)
        for i in range(self._cardinality):
            f_i = lambda x: self._eval(self._es[i], x)
            self._weights[i] = integrate(f_i, self._domain[0], self._domain[1], device=self._device)
        return
    
    def _compute_mass(self) -> None:
        """Uses numerical integration to approximate the mass matrix 
        (the integrals of the product of each pair of basis functions 
        over the domain).
        """
        self._mass = torch.zeros((self._cardinality, self._cardinality), device=self._device)
        for i in range(self._cardinality):
            for j in range(i, self._cardinality):
                e_i, e_j = self._es[i], self._es[j]
                f_ij = lambda ls: self._eval(e_i, ls) * self._eval(e_j, ls)
                integral = integrate(f_ij, self._domain[0], self._domain[1], device=self._device)
                self._mass[i, j] = self._mass[j, i] = integral
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
        dls = ls[:, None] - self._nodes
        dls = LagrangeP._adjust_dls(dls)
        sum_terms = self._omega / dls
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
    To construct a higher-order Lagrange basis, we divide the 
    approximation interval into `num_elems` equisized elements, and use 
    a set of Lagrange polynomials of degree $n=\,$`order` within each 
    element.
     
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
    Barycentric formula [@Berrut2004], which is more efficient and 
    stable than the above formula.

    We use piecewise Chebyshev polynomials of the second kind to 
    represent the (conditional) CDFs corresponding to the higher-order 
    Lagrange representation of (the square root of) the target density 
    function.
    
    """

    def __init__(
        self, 
        order: int, 
        num_elems: int, 
        device: torch.device = torch.get_default_device()
    ):

        if order == 1:
            msg = ("When 'order=1', Lagrange1 should be used " 
                   + "instead of LagrangeP.")
            raise Exception(msg)

        Piecewise.__init__(self, order, num_elems, device)
        self._local = _LagrangeRef(self._order+1, device)

        # Define Jacobian of mapping from the domain of the LagrangeRef 
        # polynomial to an element
        self._jac = self._elem_size / self._local._domain_size

        self._compute_nodes()
        self._compute_mass()
        self._compute_int_W()

        # elem_nodes[i] returns the nodes corresponding to element i
        self._elem_nodes = torch.tensor([
            range(n*self._order, (n+1)*self._order+1) 
            for n in range(self._num_elems)], device=self._device)

        return
    
    @property
    def _cardinality(self) -> int:
        return self._nodes.numel()
    
    @property
    def _domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0], device=self._device)
    
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
        n_loc = self._local._cardinality
        n_nodes = self._num_elems * (n_loc-1) + 1
        nodes = torch.zeros(n_nodes, device=self._device)
        for i in range(self._num_elems):
            inds_elem = torch.arange(n_loc, device=self._device) + i * (n_loc-1)
            nodes[inds_elem] = self._grid[i] + self._elem_size * self._local._nodes    
        self._nodes = nodes
        return
    
    def _compute_mass(self) -> None:
        """Computes the mass matrix and its Cholesky factor."""
        n_loc = self._local._cardinality
        mass_elem = self._local._mass * (0.5 * self._jac)
        self._mass = torch.zeros((self._cardinality, self._cardinality), device=self._device)
        for i in range(self._num_elems):
            inds_elem = torch.arange(n_loc, device=self._device) + i * (n_loc-1)
            self._mass[inds_elem[:, None], inds_elem[None, :]] += mass_elem
        self._mass_R = torch.linalg.cholesky(self._mass).T
        return
    
    def _compute_int_W(self) -> None:
        """Computes the integration operator."""
        n_loc = self._local._cardinality
        weights_elem = self._local._weights * (0.5 * self._jac)
        self._int_W = torch.zeros(self._cardinality, device=self._device)
        for i in range(self._num_elems):
            inds_elem = torch.arange(n_loc, device=self._device) + i * (n_loc-1)
            self._int_W[inds_elem] += weights_elem
        return

    def _eval_basis(self, ls: Tensor) -> Tensor:
        
        self._check_in_domain(ls)
        
        n_ls = ls.numel()
        ps = torch.zeros((n_ls, self._cardinality), device=self._device)
        
        left_inds = self._get_left_hand_inds(ls)
        ls_local = self._map_to_element(ls, left_inds)
        
        dls = ls_local[:, None] - self._local._nodes
        dls = self._adjust_dls(dls)
        sum_terms = self._local._omega / dls
        ps_loc = sum_terms / sum_terms.sum(1, keepdim=True)

        ii = torch.arange(n_ls, device=self._device).repeat_interleave(self._local._cardinality)
        jj = self._elem_nodes[left_inds].flatten()
        ps[ii, jj] = ps_loc.flatten()
        return ps
    
    def _eval_basis_deriv(self, ls: Tensor) -> Tensor:
        
        self._check_in_domain(ls)

        n_ls = ls.numel()
        dpdls = torch.zeros((n_ls, self._cardinality), device=self._device)
        
        left_inds = self._get_left_hand_inds(ls)
        ls_local = self._map_to_element(ls, left_inds)
        
        dls = ls_local[:, None] - self._local._nodes
        dls = self._adjust_dls(dls)
        
        sum_terms = self._local._omega / dls
        sum_terms_sq = self._local._omega / dls.square()

        coefs_b = 1.0 / torch.sum(sum_terms, dim=1, keepdim=True)
        coefs_a = torch.sum(sum_terms_sq, dim=1, keepdim=True) * coefs_b.square()

        dpdls_loc = (coefs_a * sum_terms - coefs_b * sum_terms_sq) / self._jac
        ii = torch.arange(n_ls, device=self._device).repeat_interleave(self._local._cardinality)
        jj = self._elem_nodes[left_inds].flatten()
        dpdls[ii, jj] = dpdls_loc.flatten()
        return dpdls