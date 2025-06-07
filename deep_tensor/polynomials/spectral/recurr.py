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
        norm: float | Tensor
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
        norm: 
            Normalising constant, which ensures that the norm of each 
            polynomial (with respect to the weight function) is equal 
            to 1.

        Notes
        -----
        This class needs to support 'order=0' (when forming LagrangeP 
        polynomials of degree 2).

        """
        self.order = order
        self.a = a
        self.b = b
        self.c = c
        self.norm = norm
        self._nodes, self._weights = self.compute_nodes_weights(a, b, c)
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
            {n+1}-dimensional vectors (where n denotes the order of the 
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

        self._check_in_domain(ls)
        
        ps = torch.zeros((ls.numel(), self.order+1))
        ps[:, 0] = 1.0
        if self.order == 0:
            return ps
        
        ps[:, 1] = self.a[0] * ls + self.b[0]
        for j in range(1, self.order):
            ps_jm = ps[:, j-1].clone()
            ps_j = ps[:, j].clone()
            ps[:, j+1] = (self.a[j] * ls + self.b[j]) * ps_j - self.c[j] * ps_jm
        
        return ps * self.norm
    
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:

        self._check_in_domain(ls)
        
        dpdls = torch.zeros((ls.numel(), self.order+1))
        if self.order == 0:
            return dpdls
        
        ps = self.eval_basis(ls) / self.norm
        
        dpdls[:, 1] = self.a[0] * ps[:, 0]
        for j in range(1, self.order):
            dpdls_jm = dpdls[:, j-1].clone()
            dpdls_j = dpdls[:, j].clone()
            dpdls[:, j+1] = (self.a[j] * ps[:, j] 
                             + (self.a[j] * ls + self.b[j]) * dpdls_j
                             - self.c[j] * dpdls_jm)

        return dpdls * self.norm