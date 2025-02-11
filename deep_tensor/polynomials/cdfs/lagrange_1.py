import warnings

import torch

from ..polynomials.piecewise import Piecewise


# Integrals and weights of adjacent basis functions mapped to [0, 1]
LOCAL_MASS = torch.tensor([[2.0, 1.0], [1.0, 2.0]]) / 6.0
LOCAL_WEIGHTS = torch.tensor([1.0, 1.0]) / 2.0


class Lagrange1(Piecewise):

    def __init__(self, num_elems: int):
        
        super().__init__(order=1, num_elems=num_elems)
        self._nodes = self.grid.clone()
        
        mass = torch.zeros((self.cardinality, self.cardinality))
        jac = self.elem_size / self.domain_size
        self._int_W = torch.zeros(self.cardinality)

        for i in range(self.num_elems):
            ind = torch.tensor([i, i+1])
            mass[ind[:, None], ind[None, :]] += LOCAL_MASS * jac
            self._int_W[ind] += LOCAL_WEIGHTS * jac

        self._mass_R = torch.linalg.cholesky(mass).T
        return
    
    @property
    def nodes(self) -> torch.Tensor:
        return self._nodes
    
    @property 
    def mass_R(self) -> torch.Tensor:
        return self._mass_R

    @property 
    def int_W(self) -> torch.Tensor: 
        return self._int_W

    def eval_basis(self, ls: torch.Tensor) -> torch.Tensor:

        if not torch.all(inside := self.in_domain(ls)):
            warnings.warn("Some points are outside the domain.")

        if not torch.any(inside):
            basis_vals = torch.zeros((ls.numel(), self.cardinality))
            return basis_vals
        
        inside_inds = inside.nonzero().flatten()
        left_inds = self.get_left_hand_inds(ls[inside])

        # Convert to local coordinates
        ls_local = (ls[inside]-self.grid[left_inds]) / self.elem_size

        row_inds = torch.hstack((inside_inds, inside_inds))
        col_inds = torch.hstack((left_inds, left_inds+1))
        indices = torch.vstack((row_inds, col_inds))
        vals = torch.hstack((1.0-ls_local, ls_local))

        basis_vals = torch.sparse_coo_tensor(
            indices=indices, 
            values=vals, 
            size=(ls.numel(), self.cardinality)
        )

        return basis_vals
        
    def eval_basis_deriv(self, ls: torch.Tensor) -> torch.Tensor:

        if not torch.all(inside := self.in_domain(ls)):
            warnings.warn("Some points are outside the domain.")

        if not torch.any(inside):
            deriv_vals = torch.zeros((ls.numel(), self.cardinality()))
            return deriv_vals
        
        inside_inds = inside.nonzero().flatten()
        left_inds = self.get_left_hand_inds(ls[inside])

        row_inds = torch.hstack((inside_inds, inside_inds))
        col_inds = torch.hstack((left_inds, left_inds+1))
        indices = torch.vstack((row_inds, col_inds))
        
        derivs = torch.ones_like(ls[inside]) / self.elem_size
        vals = torch.hstack((-derivs, derivs))
        
        deriv_vals = torch.sparse_coo_tensor(
            indices=indices, 
            values=vals, 
            size=(ls.numel(), self.cardinality)
        )

        return deriv_vals