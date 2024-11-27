import warnings

import torch

from .piecewise import Piecewise


class Lagrange1(Piecewise):

    def __init__(self, num_elems: int):
        
        super().__init__(order=1, num_elems=num_elems)

        self.local_mass = torch.tensor([[2.0, 1.0], [1.0, 2.0]]) / 6.0
        self.local_weights = torch.tensor([1.0, 1.0]) / 2.0
        self.local_domain = torch.tensor([0.0, 1.0])

        self._nodes = self.grid
        self.jac = self.elem_size
        
        unweighted_mass = torch.zeros((self.cardinality, self.cardinality))
        unweighted_weights = torch.zeros(self.cardinality)

        for i in range(self.num_elems):
            ind = torch.tensor([i, i+1])
            unweighted_mass[ind[:, None], ind[None, :]] += self.local_mass * self.jac
            unweighted_weights[ind] += self.local_weights * self.jac

        unweighted_mass_R = torch.linalg.cholesky(unweighted_mass).T
        
        self.mass = unweighted_mass / self.elem_size
        self._mass_R = unweighted_mass_R / self.domain_size.sqrt()
        self._int_W = unweighted_weights / self.domain_size

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
    
    def eval_basis(self, xs: torch.Tensor) -> torch.Tensor:
            
        points_in_domain = self.in_domain(xs)
        inds_in_domain = points_in_domain.nonzero().flatten()

        if not torch.all(points_in_domain):
            warnings.warn("Some points are outside the domain")

        if not torch.any(points_in_domain):
            basis_vals = torch.zeros((xs.numel(), self.cardinality))
            return basis_vals

        inside_points = xs[points_in_domain]

        left_inds = (inside_points-self.domain[0]) / self.elem_size
        left_inds = left_inds.floor().int()
        left_inds[left_inds == self.num_elems] = self.num_elems - 1

        # Convert to local coordinates
        xs_local = (inside_points-self.grid[left_inds]) / self.elem_size

        row_inds = torch.concatenate((inds_in_domain, inds_in_domain))
        col_inds = torch.concatenate((left_inds, left_inds+1))
        indices = torch.vstack((row_inds, col_inds))
        vals = torch.concatenate((1-xs_local, xs_local))

        basis_vals = torch.sparse_coo_tensor(
            indices=indices, 
            values=vals, 
            size=(xs.numel(), self.cardinality)
        )

        return basis_vals
        
    def eval_basis_deriv(self, x: torch.Tensor) -> torch.Tensor:

        # TODO: move the below ~12 lines to their own function?
        # It could be a good idea to have a separate function that 
        # returns the elements associated with each value of x
        points_in_domain = self.in_domain(x)
        inds_in_domain = points_in_domain.nonzero()

        if not torch.all(points_in_domain):
            warnings.warn("Some points are outside the domain")

        if not torch.any(points_in_domain):
            deriv_vals = torch.zeros((x.numel(), self.cardinality()))
            return deriv_vals
        
        inside_points = x[points_in_domain]
        inds = ((inside_points-self.domain[0]) / self.elem_size).floor()

        row_inds = torch.concatenate((inds_in_domain, inds_in_domain))
        col_inds = torch.concatenate((inds, inds+1))
        vals = torch.concatenate((-torch.ones_like(x), torch.ones_like(x)))
        vals /= self.elem_size
        
        deriv_vals = torch.sparse_coo_tensor(
            indices=(row_inds, col_inds), 
            values=vals, 
            size=(x.numel(), self.cardinality)
        )

        return deriv_vals