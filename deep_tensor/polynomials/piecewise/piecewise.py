import abc

import torch
from torch import Tensor

from ..basis_1d import Basis1D


class Piecewise(Basis1D, abc.ABC):

    def __init__(self, order: int, num_elems: int):
        self.order = order 
        self.num_elems = num_elems
        self.grid = torch.linspace(*self.domain, num_elems+1)
        self.elem_size = self.grid[1] - self.grid[0]
        self.domain_size = self.domain[1] - self.domain[0]
        return
    
    @property
    def domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0])
    
    @property 
    def grid(self) -> Tensor:
        return self._grid
    
    @grid.setter 
    def grid(self, value: Tensor) -> None:
        self._grid = value 
        return
    
    @property
    def num_elems(self) -> int:
        return self._num_elems
    
    @num_elems.setter 
    def num_elems(self, value: int) -> None:
        self._num_elems = value 
        return
    
    @property 
    def constant_weight(self) -> bool:
        return True

    def get_left_hand_inds(self, ls: Tensor) -> Tensor:
        """Returns the indices of the nodes that are directly to the 
        left of each of a given set of points.
        
        Parameters
        ----------
        ls:
            An n-dimensional vector of points within the local domain.

        Returns
        -------
        left_inds:
            An n-dimensional vector containing the indices of the nodes
            of the basis directly to the left of each element in ls.
        
        """

        left_inds = ((ls-self.domain[0]) / self.elem_size).floor().int()
        left_inds = left_inds.clamp(0, self.num_elems-1)
        return left_inds
    
    def map_to_element(self, ls: Tensor, left_inds: Tensor) -> Tensor:
        """Maps from a set of points in the global space to the 
        positions of the points of the elements they lie on, 
        normalising into the range [0, 1].
        """
        return (ls - self.grid[left_inds]) / self.elem_size

    def sample_measure(self, n: int) -> Tensor:
        return self.domain[0] + self.domain_size * torch.rand(n)

    def sample_measure_skip(self, n: int) -> Tensor:
        return self.sample_measure(n)

    def eval_measure(self, ls: Tensor) -> Tensor:
        return torch.full(ls.shape, 1.0 / self.domain_size)

    def eval_log_measure(self, ls: Tensor) -> Tensor:
        return torch.full(ls.shape, -self.domain_size.log())

    def eval_measure_deriv(obj, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)

    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)