import abc
import math

import torch
from torch import Tensor

from ..basis_1d import Basis1D


class Piecewise(Basis1D, abc.ABC):

    def __init__(
        self, 
        order: int, 
        num_elems: int,
        device: torch.device
    ):
        self._order = order 
        self._num_elems = num_elems
        self._device = device
        self._grid = torch.linspace(-1.0, 1.0, num_elems+1, device=self._device)
        self._elem_size = self._grid[1] - self._grid[0]
        self._domain_size = float(self._domain[1] - self._domain[0])
        return
    
    @property
    def _domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0], device=self._device)
    
    @property 
    def _constant_weight(self) -> bool:
        return True

    def _get_left_hand_inds(self, ls: Tensor) -> Tensor:
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

        left_inds = ((ls-self._domain[0]) / self._elem_size).floor().int()
        left_inds = left_inds.clamp(0, self._num_elems-1)
        return left_inds
    
    def _map_to_element(self, ls: Tensor, left_inds: Tensor) -> Tensor:
        """Maps from a set of points in the global space to the 
        positions of the points of the elements they lie on, 
        normalising into the range [0, 1].
        """
        return (ls - self._grid[left_inds]) / self._elem_size

    def _sample_measure(self, n: int) -> Tensor:
        return self._domain[0] + self._domain_size * torch.rand(n, device=self._device)

    def _eval_measure(self, ls: Tensor) -> Tensor:
        return torch.full_like(ls, 1.0 / self._domain_size)

    def _eval_log_measure(self, ls: Tensor) -> Tensor:
        return torch.full_like(ls, -math.log(self._domain_size))

    def _eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)

    def _eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)