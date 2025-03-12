import torch
from torch import Tensor

from .spectral import Spectral 
from ...constants import EPS
from ...tools import check_finite


class Chebyshev1st(Spectral):
    """Chebyshev polynomials of the first kind.

    We use the Gauss-Chebyshev set of collocation points.

    Parameters
    ----------
    order:
        The maximum order of the polynomials.
     
    References
    ----------
    Boyd, JP (2001, Appendix A.2). *[Chebyshev and Fourier spectral 
    methods](https://link.springer.com/book/9783540514879).* Lecture 
    Notes in Engineering, Volume 49.

    Cui, T, Dolgov, S and Zahm, O (2023). *[Self-reinforced polynomial 
    approximation methods for concentrated probability 
    densities.](https://arxiv.org/abs/2303.02554)*.

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

    def eval_measure(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return 1.0 / (torch.pi * ts.sqrt())
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return (ls / torch.pi) * ts.pow(-3.0/2.0)

    def eval_log_measure(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return -0.5 * ts.log() - torch.tensor(torch.pi).log()

    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        return ls / ts

    def sample_measure(self, n: int) -> Tensor:
        zs = torch.rand(n)
        samples = torch.sin(torch.pi * (zs - 0.5))
        return samples

    def sample_measure_skip(self, n: int) -> torch.Tensor:
        l0 = 0.5 * (self.nodes.min() - 1.0)
        l1 = 0.5 * (self.nodes.max() + 1.0)
        samples = l0 + torch.rand(n) * (l1 - l0)
        return samples
    
    def eval_basis(self, ls: Tensor) -> Tensor:
        thetas = self.l2theta(ls)
        thetas = thetas[:, None]
        ps = self.norm * torch.cos(thetas * self.n)
        return ps
    
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:

        if self.order == 0:
            return torch.zeros((ls.numel(), 1))

        thetas = self.l2theta(ls)
        thetas = thetas[:, None]
        sin_thetas = thetas.sin()
        sin_thetas[sin_thetas.abs() < EPS] = EPS

        dpdls = self.norm * torch.hstack((
            torch.zeros_like(thetas),
            self.n[1:] * torch.sin(thetas * self.n[1:]) / sin_thetas
        ))
        check_finite(dpdls)
        return dpdls 