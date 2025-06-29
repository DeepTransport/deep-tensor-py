from typing import Tuple
import os

import numpy as np
from scipy import optimize
import torch
from torch.autograd.functional import jacobian, hessian

from examples.plotting import pairplot

import deep_tensor as dt


def read_credit_data(fname: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reads in the German credit dataset, then shifts and scales the 
    predictors such that each has a mean of zero and standard deviation 
    of 1, and scales the response variable such that it takes values in 
    {0, 1}.
    """

    with open(fname, "r") as f:
        data = [[float(l) for l in line.strip().split()] 
                for line in f.readlines()]
    
    data = torch.tensor(data)
    xs, ys = data[:, :-1], data[:, -1]
    
    mean_xs = torch.mean(xs, dim=0)
    std_xs = torch.std(xs, dim=0)

    xs = (xs - mean_xs) / std_xs
    ys -= 1.0

    return xs, ys

fname = os.path.join("examples", "credit", "german.data-numeric")
xs, ys = read_credit_data(fname)

n_beta = 1 + xs.shape[1]

mean_pri = torch.zeros((n_beta,))
sd_pri = 10.0
cov_pri = sd_pri ** 2 * torch.eye(n_beta)

def negloglik(bs: torch.Tensor) -> torch.Tensor:

    bs = torch.atleast_2d(bs)

    neglogodds = bs[:, :1] + torch.sum(bs[:, 1:, None] * xs.T[None, ...], dim=1)
    probs = 1.0 / (1.0 + torch.exp(-neglogodds))

    neglogliks_0 = -torch.log(1.0 - probs)[:, ys < 0.5].sum(dim=1)
    neglogliks_1 = -torch.log(probs)[:, ys > 0.5].sum(dim=1)
    neglogliks = neglogliks_0 + neglogliks_1 - 500  # numerical stability
    return neglogliks

def neglogpri(bs: torch.Tensor) -> torch.Tensor:

    bs = torch.atleast_2d(bs)
    
    neglogpris = 0.5 * (bs / sd_pri).square().sum(dim=1)
    return neglogpris

def neglogpost(bs: torch.Tensor) -> torch.Tensor:
    return negloglik(bs) + neglogpri(bs)

def compute_laplace_approx() -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes a Laplace approximation to the posterior."""
    
    def jac(_bs: np.ndarray) -> torch.Tensor:
        bs = torch.from_numpy(_bs)
        return jacobian(lambda x: neglogpost(x[None, :]), bs).flatten()

    res = optimize.minimize(
        fun=lambda bs: neglogpost(torch.from_numpy(bs)), 
        x0=torch.zeros((n_beta,)),
        jac=jac
    )

    if not res.success:
        msg = "MAP optimisation failed to converge."
        raise Exception(msg)

    bs_map = torch.from_numpy(res.x)
    H = hessian(lambda x: neglogpost(x[None, :]), bs_map)
    H_inv = torch.linalg.inv(H)
    return bs_map, H_inv

bs_map, cov_map = compute_laplace_approx()

domain = dt.BoundedDomain(torch.tensor([-6.0, 6.0]))
reference = dt.GaussianReference(domain=domain)
preconditioner = dt.GaussianPreconditioner(bs_map, cov_map, reference)

bases = dt.Lagrange1(num_elems=20)

dirt = dt.DIRT(
    negloglik,
    neglogpri,
    preconditioner,
    bases,
    tt_options=dt.TTOptions(verbose=2, init_rank=10, max_rank=12)
)

n_steps = 10_000

norm = torch.distributions.MultivariateNormal(bs_map.flatten(), cov_map)
samples = norm.sample((n_steps,))
potentials_norm = -norm.log_prob(samples)
potentials_true = negloglik(samples) + neglogpri(samples)

res = dt.run_independence_sampler(samples, potentials_norm, potentials_true)
print(res.acceptance_rate)
print(res.iacts.max())
print(res.ess.min())

rs = dirt.reference.random(d=dirt.dim, n=n_steps)
samples, potentials_dirt = dirt.eval_irt(rs)
potentials_true = negloglik(samples) + neglogpri(samples)

res = dt.run_independence_sampler(samples, potentials_dirt, potentials_true)
print(res.acceptance_rate)
print(res.iacts.max())
print(res.ess.min())

rs = dirt.reference.random(d=dirt.dim, n=1000)
samples = preconditioner.Q(rs, "first")
pairplot(res.xs[::5, :6])