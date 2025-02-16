import torch
from torch import Tensor

from .spectral import Spectral 
from ...constants import EPS


class Chebyshev1st(Spectral):
    """Chebyshev polynomials of the first kind.

    We use the Gauss-Chebyshev set of collocation points.

    Parameters
    ----------
    order:
        The order of the polynomial.
     
    References
    ----------
    Boyd, JP (2001). Chebyshev and Fourier spectral methods. Appendix 
    A.2.
    https://en.wikipedia.org/wiki/Chebyshev-Gauss_quadrature

    """

    def __init__(self, order: int):

        self.order = order
        self.n = torch.arange(self.order+1)
        self.nodes = torch.cos(torch.pi * (self.n+0.5) / (self.order+1)) 
        self.nodes = self.nodes.sort()[0]
        self.weights = torch.ones_like(self.nodes) / (self.order+1)

        self.norm = torch.hstack((
            torch.tensor([1.0]), 
            torch.full((self.order,), torch.tensor(2.0).sqrt())
        ))

        self.__post_init__()
        return
    
    @property 
    def nodes(self) -> Tensor:
        return self._nodes
    
    @nodes.setter
    def nodes(self, value: Tensor) -> None:
        self._nodes = value 
        return
    
    @property 
    def domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0])
    
    @property 
    def weights(self) -> Tensor:
        return self._weights
    
    @weights.setter
    def weights(self, value: Tensor) -> None:
        self._weights = value 
        return

    @property
    def constant_weight(self) -> bool: 
        return False

    def eval_basis(self, ls: Tensor) -> Tensor:
        thetas = self.l2theta(ls)
        thetas = thetas[:, None]
        ps = torch.cos(thetas * self.n) * self.norm
        return ps
    
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:
        """Derivative of first-order polynomials is the value of 
        second-order polynomials (Wikipedia).
        """

        if self.order == 0:
            return torch.zeros_like(ls)

        thetas = self.l2theta(ls)
        thetas = thetas[:, None]

        deriv_vals = torch.hstack((
            torch.zeros_like(thetas), 
            torch.sin(thetas * self.n[1:]) * self.n[1:] / torch.sin(thetas)
        ))
        deriv_vals = deriv_vals * self.norm
        return deriv_vals 

    def eval_measure(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return 1.0 / (torch.pi * ts.sqrt())
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return (ls / torch.pi) * ts**(-3/2)

    def eval_log_measure(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return -0.5*ts.log() - torch.tensor(torch.pi).log()

    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return ls / ts

    def sample_measure(self, n: int) -> Tensor:
        zs = torch.rand(n)
        samples = torch.sin(torch.pi * (zs-0.5))
        return samples

    def sample_measure_skip(self, n: int) -> torch.Tensor:
        l0 = 0.5 * (torch.min(self.nodes) - 1.0)
        l1 = 0.5 * (torch.max(self.nodes) + 1.0)
        samples = l0 + torch.rand(n) * (l1 - l0)
        return samples

