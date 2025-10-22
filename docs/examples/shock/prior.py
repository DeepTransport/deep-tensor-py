import math
from typing import Tuple, Callable
import warnings

import torch
from torch import Tensor
from torch.distributions import Gamma


EPS = 1e-14


def regula_falsi_step(
    z0s: Tensor, 
    z1s: Tensor,
    l0s: Tensor, 
    l1s: Tensor
) -> Tuple[Tensor, Tensor]:
    """Carries out a single regula falsi iteration."""
    dls = -z1s * (l1s - l0s) / (z1s - z0s)
    dls[dls.isinf() | dls.isnan()] = 0.0
    ls = l1s + dls
    ls = torch.clamp(ls, l0s, l1s)
    return ls, dls


def regula_falsi(
    func: Callable[[Tensor], Tuple[Tensor, Tensor]],
    l0s: Tensor, 
    l1s: Tensor,
    max_iter: int = 100
) -> Tensor:
    z0s = func(l0s)[0]
    z1s = func(l1s)[0]
    for _ in range(max_iter):
        ls, dls = regula_falsi_step(z0s, z1s, l0s, l1s)
        zs = func(ls)[0]
        if converged(zs, dls):
            return ls 
        l0s[zs < 0] = ls[zs < 0]
        l1s[zs > 0] = ls[zs > 0]
        z0s[zs < 0] = zs[zs < 0]
        z1s[zs > 0] = zs[zs > 0]
    msg = (
        f"Regula Falsi failed to converge. "
        f"Maximum residual: {zs.abs().max():.2f}."
    )
    warnings.warn(msg)
    return ls


def newton_step(
    ls: Tensor,
    zs: Tensor,
    dzs: Tensor,
    l0s: Tensor, 
    l1s: Tensor 
) -> Tuple[Tensor, Tensor]:
    """Carries out a single Newton iteration."""
    dls = -zs / dzs 
    dls[dls.isinf() | dls.isnan()] = 0.0
    ls = ls + dls 
    ls = torch.clamp(ls, l0s, l1s)
    return ls, dls


def newton(
    func: Callable[[Tensor], Tuple[Tensor, Tensor]],
    l0s: Tensor,
    l1s: Tensor,
    max_iter: int = 100
) -> Tensor:
    z0s = func(l0s)[0]
    z1s = func(l1s)[0]
    ls, dls = regula_falsi_step(z0s, z1s, l0s, l1s)
    for _ in range(max_iter):  
        zs, dzs = func(ls)
        ls, dls = newton_step(ls, zs, dzs, l0s, l1s)
        if converged(zs, dls):
            return ls
    return regula_falsi(func, l0s, l1s)


def converged(fs: Tensor, dls: Tensor) -> bool:
    return fs.abs().max().item() < EPS


def gaussian_potential(xs: Tensor, mus: Tensor, sds: Tensor) -> Tensor:
    dzdxs = 0.5 * ((xs-mus)**2/sds**2) + torch.log(sds * math.sqrt(2.0*torch.pi))
    return dzdxs


def gaussian_cdf(xs: Tensor, mus: Tensor, sds: Tensor) -> Tensor:
    zs = 0.5 * (1.0 + torch.erf((xs-mus) / (sds * torch.tensor(2.0).sqrt())))
    return zs


class GammaDist():
    """A bounded Gamma density with a given rate and scale parameter.
    
    Parameters
    ----------
    alpha: 
        Scale parameter.
    lamb: 
        Rate parameter.
    bounds:
        Left- and right-hand bounds.
    
    """

    def __init__(self, alpha: Tensor, lamb: Tensor, bounds: Tensor):
        
        self.alpha = alpha
        self.lamb = lamb
        self.bounds = bounds
        self.Gamma = Gamma(self.alpha, 1.0 / self.lamb)

        self.cdf_bounds: Tensor = self.Gamma.cdf(bounds)
        self.dx = self.cdf_bounds[1] - self.cdf_bounds[0]

        self.grid = torch.linspace(*self.bounds, steps=500)
        self.cdf_grid = (self.Gamma.cdf(self.grid) - self.cdf_bounds[0]) / self.dx

        return 
    
    def eval_potential(self, xs: Tensor) -> Tensor:
        return -self.Gamma.log_prob(xs) + self.dx.log()

    def eval_cdf(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        zs = (self.Gamma.cdf(xs) - self.cdf_bounds[0]) / self.dx
        dzdxs = torch.exp(-self.eval_potential(xs))
        return zs, dzdxs
    
    def invert_cdf(self, zs_cdf: Tensor) -> Tensor:

        def func(xs: Tensor) -> Tuple[Tensor, Tensor]:
            zs, dzdxs = self.eval_cdf(xs) 
            zs -= zs_cdf
            return zs, dzdxs
        
        left_inds = ((self.cdf_grid - zs_cdf[:, None]) < 0).sum(dim=1) - 1
        l0s = self.grid[left_inds]
        l1s = self.grid[left_inds+1]
        return newton(func, l0s, l1s)
    

class GaussianDist():
    """A bounded Gamma density with a given rate and scale parameter.
    
    Parameters
    ----------
    alpha: 
        Rate parameter.
    lamb: 
        Scale parameter.
    bounds:
        Left- and right-hand bounds.
    
    """

    def __init__(self, mus: Tensor, sds: Tensor, bounds: Tensor):
        
        self.mus = mus
        self.sds = sds
        self.bounds = bounds

        self.lb_cdf = gaussian_cdf(bounds[:, 0], self.mus, self.sds)
        self.ub_cdf = gaussian_cdf(bounds[:, 1], self.mus, self.sds)

        self.dx = self.ub_cdf - self.lb_cdf
        return 
    
    def eval_potential(self, xs: Tensor) -> Tensor:
        return gaussian_potential(xs, self.mus, self.sds) + self.dx.log()

    def eval_cdf(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        zs = (gaussian_cdf(xs, self.mus, self.sds) - self.lb_cdf) / self.dx
        dzdxs = torch.exp(-self.eval_potential(xs))
        return zs, dzdxs
    
    def invert_cdf(self, zs: Tensor) -> Tensor:
        zs = (self.dx * zs) + self.lb_cdf
        zs = zs.clamp(1e-10, 1-1e-10)
        return self.mus + self.sds * math.sqrt(2.0) * torch.erfinv(2.0*zs-1.0)