import math
from typing import Callable

import torch 
from torch import Tensor 

import deep_tensor as dt


def _propose(
    a: float,
    b: float,
    xs: Tensor
) -> Tensor:
    
    xis = torch.randn_like(xs)
    rs_prop = b * xs + a * xis
    return rs_prop
    

def _eval_neglogproposal(
    a: float, 
    b: float,
    xs: Tensor, 
    xs_prop: Tensor
) -> Tensor:
    
    dim = xs.shape[1]
    mus = b * xs
    neglogproposals = (
        0.5 * dim * math.log(2.0*math.pi)
        + dim * a
        + (1.0 / (2.0*a**2)) * (xs_prop - mus).square().sum(dim=1)
    )
    return neglogproposals


def _step(
    a: float,
    b: float,
    potential: Callable[[Tensor], Tensor],
    xs_cur: Tensor,
    neglogfxs_cur: Tensor
):

    xs = torch.zeros_like(xs_cur)
    neglogfxs = torch.zeros_like(neglogfxs_cur)

    # Propose a new set of states
    xs_prop = _propose(a, b, xs_cur)
    neglogfxs_prop = potential(xs_prop)

    # Evaluate proposal density for each of the proposals
    neglogqs_prop = _eval_neglogproposal(a, b, xs_cur, xs_prop)
    neglogqs_prev = _eval_neglogproposal(a, b, xs_prop, xs_cur)

    neglogalphas = (
        neglogfxs_prop + neglogqs_prev 
        - (neglogfxs_cur + neglogqs_prop)
    )
    alphas = torch.exp(-neglogalphas)
    accepted = alphas > torch.rand_like(alphas)
    rejected = ~accepted
    
    if accepted.any():
        xs[accepted] = xs_prop[accepted].clone()
        neglogfxs[accepted] = neglogfxs_prop[accepted].clone()
    
    if rejected.any():
        xs[rejected] = xs_cur[rejected].clone()
        neglogfxs[rejected] = neglogfxs_cur[rejected].clone()

    return xs, neglogfxs, accepted


def run_pcn(
    potential: Callable[[Tensor], Tensor],
    x0s: Tensor,
    dt_: float = 2.0,
    num_steps: int = 1000,
    num_warmup: int = 0
):
    
    a = 2.0 * math.sqrt(2.0*dt_) / (2.0+dt_)
    b = (2.0-dt_) / (2.0+dt_)

    num_chains, dim = x0s.shape

    xs = torch.zeros((num_chains, num_steps, dim))
    neglogfxs = torch.zeros((num_chains, num_steps))
    num_accepts = torch.zeros((num_chains,))

    xs[:, 0, :] = x0s 
    neglogfxs[:, 0] = potential(x0s)

    for _ in range(num_warmup):
        xs[:, 0, :], neglogfxs[:, 0], _ = _step(
            a, b, potential, xs[:, 0, :], neglogfxs[:, 0]
        )

    for i in range(num_steps-1):
        xs[:, i+1, :], neglogfxs[:, i+1], accepted = _step(
            a, b, potential, xs[:, i, :], neglogfxs[:, i]
        )
        num_accepts += accepted.int()

    iacts = torch.vstack([dt.estimate_iact(xs_i) for xs_i in xs])
    acceptance_rates = num_accepts / num_steps

    return xs, neglogfxs, acceptance_rates, iacts