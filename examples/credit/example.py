import os
from typing import Tuple

import numpy as np
from scipy import optimize
import torch
from torch.autograd.functional import jacobian, hessian

import deep_tensor as dt

from examples.plotting import pairplot


# torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


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

num_beta = 1 + xs.shape[1]

mean_pri = torch.zeros((num_beta,))
sd_pri = 10.0
cov_pri = sd_pri ** 2 * torch.eye(num_beta)


def negloglik(bs: torch.Tensor) -> torch.Tensor:

    bs = torch.atleast_2d(bs)

    neglogodds = bs[:, :1] + torch.sum(bs[:, 1:, None] * xs.T[None, ...], dim=1)
    odds = torch.exp(-neglogodds)

    neglogliks_0 = -torch.log(odds / (1.0 + odds))[:, ys < 0.5].sum(dim=1)
    neglogliks_1 = -torch.log(1.0 / (1.0 + odds))[:, ys > 0.5].sum(dim=1)
    neglogliks = neglogliks_0 + neglogliks_1 - 500  # numerical stability
    return neglogliks


def neglogpri(bs: torch.Tensor) -> torch.Tensor:
    bs = torch.atleast_2d(bs)
    neglogpris = 0.5 * (bs / sd_pri).square().sum(dim=1)
    return neglogpris


def neglogpost(bs: torch.Tensor) -> torch.Tensor:
    return negloglik(bs) + neglogpri(bs)


def compute_laplace_approx(x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes a Laplace approximation to the posterior."""
    
    def jac(_bs: np.ndarray) -> torch.Tensor:
        bs = torch.from_numpy(_bs)
        return jacobian(lambda x: neglogpost(x[None, :]), bs).flatten()

    res = optimize.minimize(
        fun=lambda bs: neglogpost(torch.from_numpy(bs)), 
        x0=x0,
        jac=jac
    )

    if not res.success:
        msg = "MAP optimisation failed to converge."
        raise Exception(msg)

    bs_map = torch.from_numpy(res.x)
    H = hessian(lambda x: neglogpost(x[None, :]), bs_map)
    H_inv = torch.linalg.inv(H)
    return bs_map, H_inv


if __name__ == "__main__":

    x0 = torch.zeros(num_beta)  # I haven't found a different MAP estimate when varying the starting location
    bs_map, cov_map = compute_laplace_approx(x0)

    domain = dt.BoundedDomain(torch.tensor([-5.0, 5.0]))
    reference = dt.GaussianReference(domain=domain)
    preconditioner = dt.GaussianMapping(bs_map, cov_map, reference)

    basis = dt.Lagrange1(num_elems=30)
    bases = dt.ApproxBases(basis, reference.domain, preconditioner.dim)

    tt_options = dt.TTOptions(max_als=3, verbose=2, init_rank=1, local_tol=0.0, max_rank=12, tt_method="amen",  als_tol=0.0)
    tt = dt.TT(tt_options)
    ftt = dt.FTT(bases, tt)

    # betas = 10 ** torch.linspace(-4.0, 0.0, 10)
    # bridge = dt.Tempering(betas.tolist())

    dirt = dt.DIRT(
        neglogpost,
        preconditioner,
        ftt,
        bridge=dt.SingleLayer()
    )

    num_steps = 2000

    # kernel = dt.pCNKernel(neglogpost, dirt, dt=10.0)
    # mcmc = dt.MCMC(kernel, num_steps, num_chains=4)
    # mcmc.run()

    norm = torch.distributions.MultivariateNormal(bs_map.flatten(), cov_map)
    samples_gauss = norm.sample((num_steps,))
    potentials_norm = -norm.log_prob(samples_gauss)
    potentials_true = negloglik(samples_gauss) + neglogpri(samples_gauss)

    res = dt.run_independence_sampler(samples_gauss, potentials_norm, potentials_true)
    print(res.acceptance_rate)
    print(res.iacts.max())
    print(res.ess.min())

    rs = dirt.reference.random(d=dirt.dim, n=num_steps)
    samples, potentials_dirt = dirt.eval_irt(rs)
    potentials_true = negloglik(samples) + neglogpri(samples)

    res = dt.run_independence_sampler(samples, potentials_dirt, potentials_true)
    print(res.acceptance_rate)
    print(res.iacts.max())
    print(res.ess.min())

    for i in range(5):
        s = samples[::5, 5*i:5*(i+1)]
        s_gauss = samples_gauss[::5, 5*i:5*(i+1)]
        pairplot(s_gauss, s)

    # rs = dirt.reference.random(d=dirt.dim, n=1000)
    # samples = preconditioner.Q(rs)