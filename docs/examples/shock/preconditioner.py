import torch 
from torch import Tensor

import deep_tensor as dt

from .prior import GammaDist, GaussianDist

EPS = torch.finfo(torch.get_default_dtype()).eps


class GammaNormalMapping(dt.Preconditioner):

    def __init__(
        self, 
        reference: dt.Reference,
        bounds: Tensor, 
        alpha: Tensor, 
        gamma: Tensor, 
        ms: Tensor, 
        sds: Tensor, 
        dim: int
    ):
        
        self.reference = reference
        self.bounds = bounds 
        self.alpha = alpha 
        self.gamma = gamma
        self.ms = ms
        self.sds = sds
        self.gam = GammaDist(alpha, 1.0/gamma, bounds[-1])
        self.dim = dim
        return
    
    def check_dimensions(self, xs: Tensor) -> None:
        if xs.shape[1] != self.dim:
            msg = ("This preconditioner does not currently work for "
                   + "subsets of the parameter.")
            raise Exception(msg)
        return

    def Q(self, us: Tensor, subset: str = "first") -> Tensor:
        
        self.check_dimensions(us)
        xs = torch.zeros_like(us)
        zs = self.reference.eval_cdf(us)[0]
        zs = torch.clamp(zs, EPS, 1.0-EPS)

        xs[:, -1] = self.gam.invert_cdf(zs[:, -1])
        norm = GaussianDist(self.ms, self.sds / xs[:, -1:].sqrt(), self.bounds[:-1])
        xs[:, :-1] = norm.invert_cdf(zs[:, :-1])
        return xs

    def Q_inv(self, xs: Tensor, subset: str = "first") -> Tensor:
        
        self.check_dimensions(xs)
        zs = torch.zeros_like(xs)
        norm = GaussianDist(self.ms, self.sds / xs[:, -1:].sqrt(), self.bounds[:-1])

        zs[:, -1] = self.gam.eval_cdf(xs[:, -1])[0]
        zs[:, :-1] = norm.eval_cdf(xs[:, :-1])[0]
        us = self.reference.invert_cdf(zs)
        return us

    def neglogdet_Q(self, us: Tensor, subset: str = "first") -> Tensor:
        
        self.check_dimensions(us)
        xs = self.Q(us)
        norm = GaussianDist(self.ms, self.sds / xs[:, -1:].sqrt(), self.bounds[:-1])
        
        potential_gam = self.gam.eval_potential(xs[:, -1])
        potential_norm = norm.eval_potential(xs[:, :-1]).sum(dim=1)
        potential_ref = self.reference.eval_potential(us)[0]
        
        return potential_ref - potential_gam - potential_norm

    def neglogdet_Q_inv(self, xs: Tensor, subset: str = "first") -> Tensor:

        self.check_dimensions(xs)
        us = self.Q_inv(xs)
        norm = GaussianDist(self.ms, self.sds / xs[:, -1:].sqrt(), self.bounds[:-1])
        
        potential_gam = self.gam.eval_potential(xs[:, -1])
        potential_norm = norm.eval_potential(xs[:, :-1]).sum(dim=1)
        potential_ref = self.reference.eval_potential(us)[0]
        
        return potential_gam + potential_norm - potential_ref