import abc 

import torch

from .basis_1d import Basis1D
from ..constants import EPS


class Spectral(Basis1D, abc.ABC):

    # @property
    # @abc.abstractmethod
    # def weights(self) -> torch.Tensor:
    #     """The values of the weighting function evaluated at each 
    #     collocation point.
    #     """
    #     return

    @property
    @abc.abstractmethod
    def weights(self) -> torch.Tensor:
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
    
    def x2theta(self, xs: torch.Tensor) -> torch.Tensor:
        """Converts a set of x values (on the interval [-1, 1]) to a 
        set of theta values (theta = arccos(x)), adjusting the 
        endpoints in case of singularities.

        Parameters
        ----------
        xs: 
            Set of input points.
        
        Returns
        -------
        : 
            The corresponding set of theta values (theta = arccos(x)).
        
        """

        theta = torch.acos(xs)
        theta[torch.abs(xs+1.0) <= EPS] = torch.pi
        theta[torch.abs(xs-1.0) <= EPS] = 0.0
        return theta

    def post_construction(self):

        # See Cui and Dolgov (2022), p 1912.
        self._basis2node = self.eval_basis(self.nodes)
        self._node2basis = self.basis2node.T * self.weights

        # TODO: move this to unit tests at some point
        # print(torch.max(torch.abs(self.basis2node @ self.node2basis - torch.eye(self.basis2node.shape[0]))))
        # assert torch.max(torch.abs(self.basis2node @ self.node2basis - torch.eye(self.basis2node.shape[0]))) < 1e-2, "node2basis/basis2node constructed incorrectly."

        # Basis functions are orthonormal w.r.t weights so mass matrix 
        # is very simple
        self._mass_R = torch.eye(self.cardinality)
        self._int_W = self.basis2node * self.weights
        return