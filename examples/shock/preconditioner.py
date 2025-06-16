import torch 
from torch import Tensor

import deep_tensor as dt

from examples.shock.prior import GammaDist, GaussianDist


def construct_preconditioner(
    reference: dt.Reference,
    bounds: Tensor,
    alpha: float,
    gamma: float,
    ms: Tensor,
    sds: Tensor,
    dim: int
) -> dt.Preconditioner:

    gam = GammaDist(alpha, 1.0/gamma, bounds[-1])

    def check_dimensions(xs: Tensor) -> None:
        if xs.shape[1] != dim:
            msg = ("This preconditioner does not currently work for "
                   + "subsets of the parameter.")
            raise Exception(msg)
        return

    def Q(us: Tensor, subset: str | None = None) -> Tensor:
        
        check_dimensions(us)
        xs = torch.zeros_like(us)
        zs = reference.eval_cdf(us)[0]

        xs[:, -1] = gam.invert_cdf(zs[:, -1])
        norm = GaussianDist(ms, sds / xs[:, -1:].sqrt(), bounds[:-1])
        xs[:, :-1] = norm.invert_cdf(zs[:, :-1])
        return xs

    def Q_inv(xs: Tensor, subset: str | None = None) -> Tensor:
        
        check_dimensions(xs)
        zs = torch.zeros_like(xs)
        norm = GaussianDist(ms, sds / xs[:, -1:].sqrt(), bounds[:-1])

        zs[:, -1] = gam.eval_cdf(xs[:, -1])[0]
        zs[:, :-1] = norm.eval_cdf(xs[:, :-1])[0]
        us = reference.invert_cdf(zs)
        return us

    def neglogdet_Q(us: Tensor, subset: str | None = None) -> Tensor:
        
        check_dimensions(us)
        xs = Q(us)
        norm = GaussianDist(ms, sds / xs[:, -1:].sqrt(), bounds[:-1])
        
        potential_gam = gam.eval_potential(xs[:, -1])
        potential_norm = norm.eval_potential(xs[:, :-1]).sum(dim=1)
        potential_ref = reference.eval_potential(us)[0]
        
        return potential_ref - potential_gam - potential_norm

    def neglogdet_Q_inv(xs: Tensor, subset: str | None = None) -> Tensor:

        check_dimensions(xs)
        us = Q_inv(xs)
        norm = GaussianDist(ms, sds / xs[:, -1:].sqrt(), bounds[:-1])
        
        potential_gam = gam.eval_potential(xs[:, -1])
        potential_norm = norm.eval_potential(xs[:, :-1]).sum(dim=1)
        potential_ref = reference.eval_potential(us)[0]
        
        return potential_gam + potential_norm - potential_ref

    preconditioner = dt.Preconditioner(
        reference, 
        Q, 
        Q_inv, 
        neglogdet_Q, 
        neglogdet_Q_inv, 
        dim=dim
    )

    return preconditioner