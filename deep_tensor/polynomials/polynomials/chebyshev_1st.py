import torch
from torch import Tensor

from .spectral import Spectral 
from ...constants import EPS


class Chebyshev1st(Spectral):
    """Chebyshev polynomials of the first kind."""

    def __init__(self, order: int):

        self.order = order
        self.n = torch.arange(self.order+1)
        self.domain = torch.tensor([-1.0, 1.0])

        self._nodes = torch.cos(torch.pi * (self.n+0.5) / (self.order+1)) 
        self._nodes = torch.sort(self._nodes)[0]
        self._weights = torch.ones_like(self._nodes) / (self.order+1)

        self.normalising = torch.hstack((
            torch.tensor([1.0]), 
            torch.full((self.order,), torch.tensor(2).sqrt())
        ))

        self.__post_init__()
        return
    
    @property 
    def nodes(self) -> Tensor:
        return self._nodes
    
    @property 
    def domain(self) -> Tensor:
        return self._domain 
    
    @domain.setter
    def domain(self, value: Tensor) -> None:
        self._domain = value
        return
    
    @property 
    def weights(self) -> torch.Tensor:
        return self._weights

    @property
    def constant_weight(self) -> bool: 
        return False

    def eval_basis(self, ls: torch.Tensor) -> torch.Tensor:
        """Evaluates the set of Chebyshev polynomials of the first 
        kind, up to order n, for all inputs x in [-1, 1].
        """

        thetas = self.l2theta(ls)
        ps = torch.cos(torch.outer(thetas, self.n)) * self.normalising
        return ps
    
    def eval_basis_deriv(self, x: torch.Tensor) -> torch.Tensor:
        """Derivative of first-order polynomials is the value of 
        second-order polynomials (Wikipedia).
        """

        if self.order > 0:
            return torch.zeros_like(x)

        theta = self.l2theta(x)

        deriv_vals = torch.concat((
            torch.zeros_like(x), 
            torch.sin(torch.outer(theta, self.n[1:]) * self.n[1:]) 
                / torch.sin(theta)
        ))
        deriv_vals *= self.normalising
        return deriv_vals 

    def eval_measure(self, x: torch.Tensor) -> torch.Tensor:
        t = 1.0 - x**2
        t[t < EPS] = EPS
        return 1.0 / (torch.pi * torch.sqrt(t))
    
    def eval_measure_deriv(self, x: torch.Tensor) -> torch.Tensor:
        t = 1.0 - x**2
        t[t < EPS] = EPS
        return (x / torch.pi) * t**(-3/2)

    def eval_log_measure(self, x: torch.Tensor) -> torch.Tensor:
        t = 1.0 - x**2
        t[t < EPS] = EPS
        return -0.5*torch.log(t) - torch.log(torch.tensor(torch.pi))

    def eval_log_measure_deriv(self, x: torch.Tensor) -> torch.Tensor:
        t = 1.0 - x**2
        t[t < EPS] = EPS
        return x / t

    def sample_measure(self, n: int) -> torch.Tensor:
        z = torch.rand(n)
        samples = torch.sin(torch.pi * (z-0.5))
        return samples

    def sample_measure_skip(self, n: int) -> torch.Tensor:
        x0 = 0.5 * (torch.min(self.nodes) - 1.0)
        x1 = 0.5 * (torch.max(self.nodes) + 1.0)
        samples = x0 + torch.rand(n) * (x1-x0)
        return samples

