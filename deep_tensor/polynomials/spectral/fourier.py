import torch
from torch import Tensor

from .spectral import Spectral


class Fourier(Spectral):
    r"""Fourier polynomials.
    
    Parameters
    ----------
    order:
        The number of sine functions the basis is composed of. The 
        total number of basis functions, $N$, is equal to `2*order+2`.
    
    Notes
    -----
    The Fourier basis for the interval $[-1, 1]$, with cardinality $N$, 
    is given by
    $$
        \left\{\frac{1}{2}, \sqrt{2}\sin(\pi x), \dots, \sqrt{2}\sin(n \pi x), 
        \sqrt{2}\cos(\pi x), \dots, \sqrt{2}\cos(n \pi x), 
        \sqrt{2}\cos(N \pi x / 2)\right\},
    $$
    where $n = 1, 2, \dots, \tfrac{N}{2}-1$. 
    
    The basis functions are orthonormal with respect to the 
    (normalised) weight function given by
    $$\lambda(x) = \frac{1}{2}.$$

    A given function can be represented in the Fourier basis as 
    $$
        f(x) \approx 
            a_{0} 
            + \sum_{n=1}^{N/2}a_{n} \sqrt{2} \cos(n \pi x) 
            + \sum_{n=1}^{N/2-1}b_{n} \sqrt{2} \sin(n \pi x),
    $$
    where the coefficients can be computed using the trapezoidal rule:
    $$
        \begin{align}
            a_{0} &= \frac{1}{N}\sum_{k=1}^{N} f(x_{k}), && \\
            a_{n} &= \frac{1}{N}\sum_{k=1}^{N} f(x_{k}) \sqrt{2} \cos(n \pi x_{k}), 
                &&\quad n = 1, 2, \dots, \frac{N}{2} - 1, \\
            b_{n} &= \frac{1}{N}\sum_{k=1}^{N} f(x_{k}) \sqrt{2} \sin(n \pi x_{k}),
                &&\quad n = 1, 2, \dots, \frac{N}{2} - 1, \\
            a_{N/2} &= \frac{1}{2N}\sum_{k=1}^{N} f(x_{k}) \sqrt{2} \cos(N \pi x_{k} / 2).
                &&
        \end{align}
    $$
    The collocation points $\{x_{n}\}_{n=1}^{N}$ are given by
    $$
        x_{n} = -1 + \frac{2n}{N}.
    $$

    References
    ----------
    Boyd, JP (2001, Section 4.5). *[Chebyshev and Fourier spectral 
    methods](https://link.springer.com/book/9783540514879).* Lecture 
    Notes in Engineering, Volume 49.

    Cui, T and Dolgov, S (2022). *[Deep composition of Tensor-Trains 
    using squared inverse Rosenblatt transports](https://doi.org/10.1007/s10208-021-09537-5).* 
    Foundations of Computational Mathematics, **22**, 1863--1922.
        
    """

    def __init__(self, order: int):
        
        n_nodes = 2 * order + 2
        n = torch.arange(n_nodes)

        self.order = order
        self.m = order + 1
        self.c = torch.pi * (torch.arange(order) + 1)
        self.nodes = 2.0 * (n+1) / n_nodes - 1
        self.weights = torch.ones_like(self.nodes) / n_nodes

        self.__post_init__()
        # TODO: figure out what's going on here
        self.node2basis[-1] *= 0.5
        return

    @property
    def domain(self) -> Tensor:
        return torch.tensor([-1.0, 1.0])
    
    @property
    def constant_weight(self) -> bool:
        return True
    
    @property 
    def nodes(self) -> Tensor:
        return self._nodes
    
    @nodes.setter 
    def nodes(self, value: Tensor) -> None:
        self._nodes = value 
        return

    @property
    def weights(self) -> Tensor:
        return self._weights
    
    @weights.setter 
    def weights(self, value: Tensor) -> None:
        self._weights = value 
        return

    def sample_measure(self, n: int) -> Tensor:
        return torch.rand(n) * 2 - 1
    
    def sample_measure_skip(self, n: int) -> Tensor:
        return self.sample_measure(n)
    
    def eval_measure(self, ls: Tensor):
        return torch.full(ls.shape, 0.5)
    
    def eval_log_measure(self, ls: Tensor) -> Tensor:
        return torch.full(ls.shape, torch.tensor(0.5).log())
    
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)
    
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        return torch.zeros_like(ls)
    
    def eval_basis(self, ls: Tensor) -> Tensor:
        ls = ls[:, None]
        ps = torch.hstack((
            torch.ones_like(ls),
            2 ** 0.5 * torch.sin(ls * self.c),
            2 ** 0.5 * torch.cos(ls * self.c),
            2 ** 0.5 * torch.cos(ls * self.m * torch.pi)
        ))
        return ps
    
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:
        ls = ls[:, None]
        dpdls = torch.hstack((
            torch.zeros_like(ls),
            2 ** 0.5 * torch.cos(ls * self.c) * self.c,
            -2 ** 0.5 * torch.sin(ls * self.c) * self.c,
            -2 ** 0.5 * torch.sin(ls * self.m * torch.pi) * self.m * torch.pi
        ))
        return dpdls