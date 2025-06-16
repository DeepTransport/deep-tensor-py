import torch
from torch import Tensor

from .piecewise import Piecewise


# Integrals and weights of adjacent basis functions mapped to [0, 1]
LOCAL_MASS = torch.tensor([[2.0, 1.0], [1.0, 2.0]]) / 6.0
LOCAL_WEIGHTS = torch.tensor([1.0, 1.0]) / 2.0


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

    def __init__(self, num_elems: int):
        
        Piecewise.__init__(self, order=1, num_elems=num_elems)
        self.nodes = self.grid.clone()
        
        mass = torch.zeros((self.cardinality, self.cardinality))
        jac = self.elem_size / self.domain_size
        self.int_W = torch.zeros(self.cardinality)

        for i in range(self.num_elems):
            ind = torch.tensor([i, i+1])
            mass[ind[:, None], ind[None, :]] += LOCAL_MASS * jac
            self.int_W[ind] += LOCAL_WEIGHTS * jac

        self.mass_R = torch.linalg.cholesky(mass).T
        return
    
    # @property
    # def nodes(self) -> Tensor:
    #     return self._nodes
    
    # @nodes.setter 
    # def nodes(self, value: Tensor) -> None:
    #     self._nodes = value 
    #     return
    
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

    def eval_basis(self, ls: Tensor) -> Tensor:
        
        self._check_in_domain(ls)
        
        inds = torch.arange(ls.numel())
        left_inds = self.get_left_hand_inds(ls)

        # Convert to local coordinates
        ls_local = (ls-self.grid[left_inds]) / self.elem_size

        ii = torch.hstack((inds, inds))
        jj = torch.hstack((left_inds, left_inds+1))
        vals = torch.hstack((1.0-ls_local, ls_local))
        ps = torch.zeros((ls.numel(), self.cardinality))
        ps[ii, jj] = vals
        return ps
    
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:

        self._check_in_domain(ls)
        
        inds = torch.arange(ls.numel())
        left_inds = self.get_left_hand_inds(ls)

        ii = torch.hstack((inds, inds))
        jj = torch.hstack((left_inds, left_inds+1))
        derivs = torch.ones_like(ls) / self.elem_size
        vals = torch.hstack((-derivs, derivs))
        dpdls = torch.zeros((ls.numel(), self.cardinality))
        dpdls[ii, jj] = vals
        return dpdls