import abc

import torch

from .spectral import Spectral


class Recurr(Spectral, abc.ABC):

    def __init__(
        self, 
        order: int,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        normalising_const: float
    ):
        
        self.a = a 
        self.b = b 
        self.c = c

        T0 = -self.b / self.a
        J1 = torch.sqrt(c[1:]/(a[:-1]*a[1:]))
        J = torch.diag(T0) + torch.diag(J1, -1) + torch.diag(J1, 1)
        eigvals, eigvecs = torch.linalg.eigh(J)

        self.order = order 
        self._nodes = eigvals
        self._weights = eigvecs[0] ** 2
        self.normalising_const = normalising_const
        
        self.post_construction()
        return

    def eval_basis(self, x: torch.Tensor) -> torch.Tensor:

        if self.order == 0:
            return torch.full(x.shape, self.normalising_const)
        
        basis = torch.zeros((x.numel(), self.order+1))
        
        basis[:, 0] = 1.0
        basis[:, 1] = self.a[0] * x + self.b[0]
        for j in range(1, self.order):
            basis[:, j+1] = ((self.a[j] * x + self.b[j]) * basis[:, j].clone() 
                             - self.c[j] * basis[:, j-1].clone())
        
        return basis * self.normalising_const
        
    def eval_basis_deriv(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.order == 0:
            return torch.zeros_like(x)
        
        basis = self.eval_basis(x)

        basis_deriv = torch.zeros((x.numel(), self.order+1))
        
        basis_deriv[:, 0] = 0.0
        basis_deriv[:, 1] = self.a[0] * basis[:, 1]

        for j in range(1, self.order):
            
            basis_deriv[:, j+1] = (
                self.a[j] * basis[:, j] 
                + (self.a[j] * x + self.b[j] * basis_deriv[:, j])
                - self.c[j] * basis_deriv[:, j-1]
            )

        return basis_deriv * self.normalising_const

