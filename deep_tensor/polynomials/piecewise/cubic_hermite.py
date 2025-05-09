import torch 
from torch import Tensor

from .piecewise import Piecewise


class CubicHermite(Piecewise):
    r"""Cubic Hermite polynomials.
    
    TODO: finish this docstring.

    """

    def __init__(self, num_elems: int):

        Piecewise.__init__(self, order=3, num_elems=num_elems)

        self.nodes = self.grid.clone()
        self.jac = self.elem_size

        self.mass_R = None  # TODO: fix this (could do numerical integration or compute integrals by hand)
        self.int_W = None
        
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

    @property 
    def int_W(self) -> Tensor: 
        return self._int_W
    
    @int_W.setter 
    def int_W(self, value: Tensor) -> None:
        self._int_W = value 
        return
    
    def _compute_mass(self) -> Tensor:
        """Computes the mass matrix and its Cholesky factor."""
        raise NotImplementedError()
        n_loc = self.local.cardinality
        mass_elem = self.local.mass * (0.5 * self.jac)
        self.mass = torch.zeros((self.cardinality, self.cardinality))
        for i in range(self.num_elems):
            inds_elem = torch.arange(n_loc) + i * (n_loc-1)
            self.mass[inds_elem[:, None], inds_elem[None, :]] += mass_elem
        self.mass_R = torch.linalg.cholesky(self.mass).T
        return
    
    def eval_basis(self, ls: Tensor) -> Tensor:
        
        self._check_in_domain(ls)
        
        n_ls = ls.numel()
        ps = torch.zeros((n_ls, 2 * self.cardinality))
        
        left_inds = self.get_left_hand_inds(ls)
        ls_local = self.map_to_element(ls, left_inds)

        V = torch.vstack((ls_local ** 3, ls_local ** 2, ls_local))

        coeffs = torch.tensor([
            [2.0, -3.0, 0.0],  # h00
            [self.elem_size, -2.0 * self.elem_size, self.elem_size],  # h01 
            [-2.0, 3.0, 0.0],  # h10
            [self.elem_size, -self.elem_size, 0.0]   # h11
        ])

        ps_loc = coeffs @ V + torch.tensor([[1.0], [0.0], [0.0], [0.0]])

        ii = torch.arange(n_ls).repeat(4)
        jj = torch.vstack((
            2 * left_inds, 
            2 * left_inds + 1,
            2 * left_inds + 2,
            2 * left_inds + 3
        )).flatten()
        ps[ii, jj] = ps_loc.flatten()
        return ps
    
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:

        raise NotImplementedError()
        # self._check_in_domain(ls)
        
        # inds = torch.arange(ls.numel())
        # left_inds = self.get_left_hand_inds(ls)

        # ii = torch.hstack((inds, inds))
        # jj = torch.hstack((left_inds, left_inds+1))
        # derivs = torch.ones_like(ls) / self.elem_size
        # vals = torch.hstack((-derivs, derivs))
        # dpdls = torch.zeros((ls.numel(), self.cardinality))
        # dpdls[ii, jj] = vals
        # return dpdls
