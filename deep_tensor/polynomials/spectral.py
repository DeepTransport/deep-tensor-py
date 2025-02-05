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
    
    @basis2node.setter
    def basis2node(self, value: torch.Tensor) -> None:
        self._basis2node = value 
        return

    @property 
    def node2basis(self) -> torch.Tensor:
        """The inverse of basis2node."""
        return self._node2basis
    
    @node2basis.setter 
    def node2basis(self, value: torch.Tensor) -> None:
        self._node2basis = value 
        return
    
    @property 
    def mass_R(self) -> torch.Tensor:
        return self._mass_R 
    
    @mass_R.setter
    def mass_R(self, value: torch.Tensor) -> None: 
        self._mass_R = value
        return
    
    @property 
    def int_W(self) -> torch.Tensor: 
        return self._int_W
    
    @int_W.setter 
    def int_W(self, value: torch.Tensor) -> None:
        self._int_W = value
        return 
    
    def x2theta(self, ls: torch.Tensor) -> torch.Tensor:
        """Converts a set of x values (on the interval [-1, 1]) to a 
        set of theta values (theta = arccos(x)), adjusting the 
        endpoints in case of singularities.

        Parameters
        ----------
        ls: 
            An n-dimensional vector containing a set of points from the 
            local domain.
        
        Returns
        -------
        thetas: 
            An n-dimensional vector containing the corresponding theta 
            values (theta = arccos(x)).
        
        """

        thetas = torch.acos(ls)
        thetas[torch.abs(ls + 1.0) <= EPS] = torch.pi
        thetas[torch.abs(ls - 1.0) <= EPS] = 0.0
        return thetas

    def post_construction(self):

        # See Cui and Dolgov (2022), p 1912.
        self._basis2node = self.eval_basis(self.nodes)
        self._node2basis = self.basis2node.T * self.weights
        
        # Basis functions are orthonormal w.r.t weights so mass matrix 
        # is very simple
        self._mass_R = torch.eye(self.cardinality)
        self._int_W = self.basis2node * self.weights
        return