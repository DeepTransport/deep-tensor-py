import torch 
from torch import Tensor
from torch.distributions.beta import Beta

from .spectral import Spectral
from ...constants import EPS
from ...tools import check_finite


class Chebyshev2nd(Spectral):
    r"""Chebyshev polynomials of the second kind.
    
    Parameters
    ----------
    order:
        The maximum order of the polynomials.

    Notes
    -----
    The (normalised) Chebyshev polynomials of the second kind, defined 
    on $(-1, 1)$, are given by
    $$
        p_{k}(x) = \frac{\sin((k+1)\arccos(x))}{\sin{(\arccos(x))}}, 
            \qquad k = 0, 1, \dots, n.
    $$
    The polynomials are orthogonal with respect to the (normalised) 
    weighting function given by
    $$
        \lambda(x) = \frac{2\sqrt{1-x^{2}}}{\pi}.
    $$

    References
    ----------
    Boyd, JP (2001, Appendix A.2). *[Chebyshev and Fourier spectral 
    methods](https://link.springer.com/book/9783540514879).* Lecture 
    Notes in Engineering, Volume 49.
        
    """

    def __init__(self, order: int):

        n = order + 1

        self.order = order 
        self.nodes = torch.cos(torch.pi * torch.arange(1, n+1) / (n+1)).sort()[0]
        self.weights = torch.sin(torch.pi * torch.arange(1, n+1) / (n+1)).square() * 2 / (n+1)
        
        self.n = torch.arange(self.order+1)
        self.norm = 1.0

        self.__post_init__()
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
    
    def sample_measure(self, n: int) -> Tensor:
        ls = Beta(1.5, 1.5).sample((n,))
        return ls
    
    def sample_measure_skip(self, n: int) -> Tensor:
        left = 0.5 * (self.nodes.min() - 1.0)
        right = 0.5 * (self.nodes.max() + 1.0)
        ls = left + torch.rand(n) * (right - left)
        return ls
    
    def eval_measure(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ts = 1.0 - ls.square()
        ws = 2.0 * ts.sqrt() / torch.pi 
        return ws
    
    def eval_log_measure(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ts = 1.0 - ls.square()
        logws = 0.5 * ts.log() + torch.tensor(2.0/torch.pi).log()
        return logws
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ls[ls < EPS] = EPS
        ts = 1.0 / (1.0 - ls.square())
        check_finite(ts)
        dwdls = -2.0 * ls * ts.sqrt() / torch.pi
        return dwdls
    
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        self._check_in_domain(ls)
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS
        ws = -ls / ts
        check_finite(ws)
        return ws
    
    def eval_basis(self, ls: Tensor) -> Tensor:

        self._check_in_domain(ls)
        
        thetas = self.l2theta(ls)[:, None]
        sin_thetas = thetas.sin()
        sin_thetas[sin_thetas.abs() < EPS] = EPS

        ps = self.norm * torch.sin(thetas * (self.n+1)) / sin_thetas

        # Deal with endpoints
        mask_lhs = (ls + 1.0).abs() < EPS
        mask_rhs = (ls - 1.0).abs() < EPS
        ps[mask_lhs] = self.norm * (self.n+1) * torch.tensor(-1.0).pow(self.n)
        ps[mask_rhs] = self.norm * (self.n+1)
        check_finite(ps)
        return ps
    
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:

        self._check_in_domain(ls)

        thetas = self.l2theta(ls)[:, None]
        sin_thetas = thetas.sin()
        sin_thetas[sin_thetas.abs() < EPS] = EPS
        ls = ls[:, None]
        
        ts = ls.square() - 1.0
        ts[ts > -EPS] = -EPS

        dpdls = self.norm * ((torch.cos(thetas * (self.n+1)) * (self.n+1)
                              - torch.sin(thetas * (self.n+1)) * (ls / sin_thetas)) / ts)
        check_finite(dpdls)
        return dpdls