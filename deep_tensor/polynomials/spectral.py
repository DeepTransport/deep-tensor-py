import abc 
from typing import Tuple

import torch

from .basis_1d import Basis1D


class Spectral(Basis1D, abc.ABC):

    @property
    @abc.abstractmethod
    def weights(self) -> torch.Tensor:
        """The values of the weighting function evaluated at each 
        collocation point.
        """
        return

    @property 
    def basis2node(self) -> torch.Tensor:
        """The values of each basis function evaluated at each 
        collocation point.
        """
        return self._basis2node

    @property 
    def node2basis(self) -> torch.Tensor:
        """The inverse of basis2node."""
        return self._node2basis
    
    @property 
    def mass_R(self) -> torch.Tensor:
        return self._mass_R 
    
    @property 
    def int_W(self) -> torch.Tensor: 
        return self._int_W

    def post_construction(self):

        # See Cui and Dolgov (2022), p 1912.
        self._basis2node = self.eval_basis(self.nodes)
        self._node2basis = self.basis2node.T * self.weights

        # TODO: move this to unit tests at some point
        assert torch.max(torch.abs(self.basis2node @ self.node2basis - torch.eye(self.basis2node.shape[0]))) < 1e-5, "node2basis/basis2node constructed incorrectly."

        # Basis functions are orthonormal w.r.t weights so mass matrix 
        # is very simple
        self._mass_R = torch.eye(self.cardinality)
        self._int_W = self.basis2node * self.weights
        return