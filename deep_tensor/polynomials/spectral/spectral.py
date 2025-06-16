import abc 

import torch
from torch import Tensor

from ..basis_1d import Basis1D
from ...tools import check_finite


class Spectral(Basis1D, abc.ABC):

    def __post_init__(self) -> None:
        """Forms the basis2node and node2basis operators, the mass 
        matrix, and the integration matrix for a given basis.
        """
        self.basis2node = self.eval_basis(self.nodes)
        self.node2basis = self.basis2node.T * self.weights
        self.omegas = self.eval_measure(self.nodes)
        self.mass_R = torch.eye(self.cardinality)
        # TODO: check this (not currently used)
        # self.int_W = self.basis2node.T @ self.weights
        self.int_W = self.basis2node * self.weights
        return

    @property
    @abc.abstractmethod
    def weights(self) -> Tensor:
        """The collocation weights.
        """
        pass

    @property 
    def basis2node(self) -> Tensor:
        """The values of each basis function evaluated at each 
        collocation point. Given a set of coefficients for each basis 
        function, returns the value of the function of interest at 
        each collocation point.
        """
        return self._basis2node
    
    @basis2node.setter
    def basis2node(self, value: Tensor) -> None:
        self._basis2node = value 
        return

    @property 
    def node2basis(self) -> Tensor:
        """The inverse of basis2node. Given the values of the function 
        of interest at each collocation point, returns the 
        correpsonding coefficients of each basis function.
        """
        return self._node2basis
    
    @node2basis.setter 
    def node2basis(self, value: Tensor) -> None:
        self._node2basis = value 
        return
    
    @property 
    def mass_R(self) -> Tensor:
        """A matrix containing the inner products of each pair of basis 
        functions, weighted by the weighting function.
        """
        return self._mass_R 
    
    @mass_R.setter
    def mass_R(self, value: Tensor) -> None: 
        self._mass_R = value
        return
    
    @property 
    def int_W(self) -> Tensor: 
        """Given a set of polynomial coefficients, this operator 
        returns the values of the integrated function at each 
        collocation point.
        """
        return self._int_W
    
    @int_W.setter 
    def int_W(self, value: Tensor) -> None:
        self._int_W = value
        return 
    
    @staticmethod
    def l2theta(ls: Tensor) -> Tensor:
        """Converts a set of values from a local domain to a set of 
        theta values (theta = arccos(l)).

        Parameters
        ----------
        ls: 
            An n-dimensional vector containing a set of points from the 
            local domain.
        
        Returns
        -------
        thetas: 
            An n-dimensional vector containing the corresponding values 
            of theta (theta = arccos(l)).
        
        """
        thetas = ls.clamp(-1.0, 1.0).acos()
        check_finite(thetas)
        return thetas