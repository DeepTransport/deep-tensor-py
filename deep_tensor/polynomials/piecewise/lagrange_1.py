import torch
from torch import Tensor

from .piecewise import Piecewise


# Integrals of adjacent basis functions mapped to [0, 1]
LOCAL_MASS = torch.tensor([[2.0, 1.0], [1.0, 2.0]]) / 6.0


class Lagrange1(Piecewise):
    r"""Piecewise linear polynomials.
    
    Parameters
    ----------
    num_elems:
        The number of elements to use.

    Notes
    -----
    To construct a piecewise linear basis, we divide the interval 
    $[0, 1]$ into `num_elems` equisized elements. Then, within each 
    element a given function can be represented by
    $$
        f(x) \approx f(x_{0}) 
            + \frac{f(x_{1}) - f(x_{0})}{x_{1} - x_{0}}(x - x_{0}),
    $$
    where $x_{0}$ and $x_{1}$ denote the endpoints of the element.

    We use piecewise cubic polynomials to represent the (conditional) 
    CDFs corresponding to the piecewise linear representation of (the 
    square root of) the target density function.
    
    """

    def __init__(
        self, 
        num_elems: int, 
        device: torch.device = torch.device("cpu")
    ):
        
        order = 1
        Piecewise.__init__(self, order, num_elems, device)
        self.nodes = self.grid.clone()
        
        jac = self.elem_size / self.domain_size
        local_mass = LOCAL_MASS.to(self.device)
        mass = self._build_mass_matrix(self.num_elems, jac, local_mass)
        self.mass_R = torch.linalg.cholesky(mass).T

        return
    
    @property 
    def mass_R(self) -> Tensor:
        return self._mass_R
    
    @mass_R.setter 
    def mass_R(self, value: Tensor) -> None:
        self._mass_R = value 
        return

    def _build_mass_matrix(
        self,
        num_elems: int, 
        jac: Tensor, 
        local_mass: Tensor
    ) -> Tensor:
        """Constructs the mass matrix for a piecewise linear basis."""
        M = torch.zeros((num_elems+1, num_elems+1), device=self.device)
        for i in range(num_elems):
            inds = torch.tensor([i, i+1], device=self.device)
            M[inds[:, None], inds[None, :]] += local_mass * jac
        return M

    def eval_basis(self, ls: Tensor) -> Tensor:
        
        ls = ls.to(self.device)
        self._check_in_domain(ls)
        
        inds = torch.arange(ls.numel(), device=self.device)
        left_inds = self.get_left_hand_inds(ls)
        # Map to local (element) coordinates
        ls_local = (ls-self.grid[left_inds]) / self.elem_size

        ii = torch.hstack((inds, inds))
        jj = torch.hstack((left_inds, left_inds+1))
        vals = torch.hstack((1.0-ls_local, ls_local))
        ps_shape = (ls.numel(), self.cardinality)
        ps = torch.zeros(ps_shape, device=self.device)
        ps[ii, jj] = vals
        
        return ps
    
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:

        ls = ls.to(self.device)
        self._check_in_domain(ls)
        
        inds = torch.arange(ls.numel(), device=self.device)
        left_inds = self.get_left_hand_inds(ls)

        ii = torch.hstack((inds, inds))
        jj = torch.hstack((left_inds, left_inds+1))
        derivs = torch.ones_like(ls) / self.elem_size
        vals = torch.hstack((-derivs, derivs))

        dpdls_shape = (ls.numel(), self.cardinality)
        dpdls = torch.zeros(dpdls_shape, device=self.device)
        dpdls[ii, jj] = vals
        
        return dpdls