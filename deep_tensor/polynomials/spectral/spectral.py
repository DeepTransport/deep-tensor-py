import abc 

import torch
from torch import Tensor

from ..basis_1d import Basis1D
from ...tools import check_finite


class Spectral(Basis1D, abc.ABC):
    _weights: Tensor

    def __post_init__(self, device: torch.device) -> None:
        """Forms the basis2node and node2basis operators, the 
        quadrature weights and the mass matrix for a given basis.
        """
        self._device = device
        self._basis2node = self._eval_basis(self._nodes)
        self._node2basis = self._basis2node.T * self._weights
        self._omegas = self._eval_measure(self._nodes)
        self._mass_R = torch.eye(self._cardinality, device=self._device)
        return

    @staticmethod
    def _l2theta(ls: Tensor) -> Tensor:
        """Applies the mapping l -> arccos(l) to a vector of values on
        [-1, 1].
        """
        thetas = ls.clone().clamp(-1.0, 1.0).arccos()
        check_finite(thetas)
        return thetas