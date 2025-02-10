import abc
from typing import Tuple

import torch
from torch import Tensor

from .spectral import Spectral


class Recurr(Spectral, abc.ABC):

    def __init__(
        self, 
        order: int,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        normalising_const: float
    ):
        """Class for spectral polynomials for which the three-term 
        recurrence relation is known. This relation takes the form

        p_{j}(x) = (a_{j}x + b_{j})p_{j-1}(x) - c_{j}p_{j-2}(x),

        where a_{j}, b_{j} and c_{j} are constants.

        Parameters
        ----------
        order:
            The maximum degree of the polynomials.
        a, b, c:
            n-dimensional vectors (where n denotes the order of the 
            polynomial) containing the coefficients of the recurrence 
            relation for the polynomial.

        """

        self.order = order         
        self.a = a 
        self.b = b 
        self.c = c
        self._nodes, self._weights = self.compute_nodes_weights(a, b, c)
        self.normalising_const = normalising_const

        self.__post_init__()
        return

    @staticmethod
    def compute_nodes_weights(
        a: Tensor,
        b: Tensor,
        c: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Computes the collocation points and the interpolation 
        weights using the Golub-Welsch method.

        Parameters
        ----------
        a, b, c:
            n-dimensional vectors (where n denotes the order of the 
            polynomial) containing the coefficients of the recurrence 
            relation for the polynomial.

        Returns
        -------
        nodes:
            An n-dimensional vector containing the collocation points 
            for the polynomial.
        weights:
            An n-dimensional vector containing the corresponding 
            interpolation weights.
        
        References
        ----------
        Golub, GH and Welsch, JH (1969). Calculation of Gauss 
        quadrature rules.

        """
        
        # Build tridiagonal matrix
        alpha = -b / a
        beta = torch.sqrt(c[1:] / (a[:-1] * a[1:]))
        J = torch.diag(alpha) + torch.diag(beta, -1) + torch.diag(beta, 1)

        eigvals, eigvecs = torch.linalg.eigh(J)
        weights = eigvecs[0] ** 2
        return eigvals, weights

    def eval_basis(self, ls: Tensor) -> Tensor:

        if self.order == 0:
            return torch.full((ls.numel(), 1), self.normalising_const)
        
        ps = torch.zeros((ls.numel(), self.order+1))
        
        # Compute first two terms in recurrence relation
        ps[:, 0] = 1.0
        ps[:, 1] = self.a[0] * ls + self.b[0]

        # Compute remaining terms
        for j in range(1, self.order):
            ps[:, j+1] = ((self.a[j] * ls + self.b[j]) * ps[:, j].clone() 
                          - self.c[j] * ps[:, j-1].clone())
        
        return ps * self.normalising_const
        
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:
        
        if self.order == 0:
            return torch.full((ls.numel(), 1), 0.0)
        
        ps = self.eval_basis(ls)

        dpdxs = torch.zeros((ls.numel(), self.order+1))
        
        # Compute first two terms in recurrence relation
        dpdxs[:, 0] = 0.0
        dpdxs[:, 1] = self.a[0] * ps[:, 1]

        # Compute remaining terms
        for j in range(1, self.order):
            dpdxs[:, j+1] = (self.a[j] * ps[:, j] 
                             + (self.a[j] * ls + self.b[j]) * dpdxs[:, j]
                             - self.c[j] * dpdxs[:, j-1])

        return dpdxs * self.normalising_const

