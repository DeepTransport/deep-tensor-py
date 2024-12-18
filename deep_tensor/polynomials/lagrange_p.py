import warnings

import torch

from .jacobi_11 import Jacobi11
from .piecewise import Piecewise
from ..constants import EPS


class LagrangeRef():
    
    def __init__(self, n: int):
        """Defines the reference Lagrange basis, in the reference
        domain [0, 1].

        Parameters
        ----------
        n: 
            The number of interpolation points to use.

        References
        ----------
        Berrut, J and Trefethen, LN (2004). Barycentric Lagrange 
        interpolation.

        """

        self.cardinality = n

        if n < 2: 
            msg = ("More than two points are needed " 
                   + "to define Lagrange interpolation.")
            raise Exception(msg)
        
        self.nodes = torch.zeros(n)
        self.nodes[-1] = 1.0

        if n > 2:
            order = n-3
            jacobi = Jacobi11(order)
            self.nodes[1:-1] = 0.5 * (jacobi.nodes + 1.0)

        # Compute the local Barycentric weights (see Berrut and 
        # Trefethen, Eq. (3.2))
        self.omega = torch.zeros(n)
        for j in range(n):
            mask = torch.full((n, ), True)
            mask[j] = False
            self.omega[j] = 1.0 / (self.nodes[j]-self.nodes[mask]).prod()
        
        # Define the mass matrix
        I = torch.eye(n)
        self.mass = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                raise NotImplementedError("TODO: finish")
                f_ij = 1.0  # TODO: finish (need to integrate)

        # Set up the integration of each basis
        self.weights = torch.zeros((n,))
        for i in range(n):
            f_i = 1.0  # TODO: finish (need to integrate)

        return

    def eval(
        self, 
        f_x: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""

        m = x.numel()
        n = self.cardinality
        f = torch.zeros((m, ))

        mask_outside = torch.bitwise_or(
            self.nodes[0] - EPS >= x,
            self.nodes[-1] + EPS <= x
        )

        if mask_outside.any():
            msg = "Points outside of domain."
            warnings.warn(msg)
            f[mask_outside] = 0.0

        x_inside = x[~mask_outside]

        raise NotImplementedError("TODO: finish")
        
        return f


class LagrangeP(Piecewise):

    def __init__(self, order, num_elems):

        Piecewise.__init__(self, order, num_elems)

        if order == 1:
            msg = ("When `order=1`, Lagrange1 should be used " 
                   + "instead of LagrangeP.")
            raise Exception(msg)
        
        self.local = LagrangeRef(self.order + 1)

        # Set up global nodes
        num_nodes = self.num_elements * (self.local.cardinality - 1) + 1

        return