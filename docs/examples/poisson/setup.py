from typing import Tuple

import dolfin as dl
import torch 

from examples.priors import ProcessConvolutionPrior
from .solver import PoissonSolver


def setup_poisson_problem() -> Tuple[ProcessConvolutionPrior, PoissonSolver]:

    mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(1.0, 1.0), 32, 32)
    V = dl.FunctionSpace(mesh, "Lagrange", 1)

    xs = torch.from_numpy(V.tabulate_dof_coordinates())

    s0s = torch.tensor([0.2, 0.5, 0.8])
    s1s = torch.tensor([0.2, 0.5, 0.8])
    ss = torch.tensor([[s0, s1] for s0 in s0s for s1 in s1s])
    mu_prior = 0.0
    r_prior = 0.05

    prior = ProcessConvolutionPrior(xs, ss, mu_prior, r_prior)

    x0_obs = torch.tensor([0.3, 0.7])
    x1_obs = torch.tensor([0.3, 0.7])
    xs_obs = torch.tensor([[x0, x1] for x0 in x0_obs for x1 in x1_obs])

    model = PoissonSolver(mesh, V, prior, xs_obs)
    return prior, model