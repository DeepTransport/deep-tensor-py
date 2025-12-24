import math
from typing import Tuple

import dolfin as dl
import numpy as np
import torch 

from examples.priors import Prior, SquaredExponential, GaussianRandomField
from examples.poisson.solver import PoissonSolver, PoissonSolverROM
from examples.poisson.reduced_order_modelling import compute_pod_basis


def setup_poisson_problem(
    rom: bool = False,
    num_kl: int | None = None
) -> Tuple[Prior, PoissonSolver | PoissonSolverROM]:

    def boundary_left(x: np.ndarray, on_boundary: bool) -> bool:
        """Returns True if the input point is on the left-hand boundary, 
        and False otherwise.
        """
        return on_boundary and (x[0] < 1e-8)

    def boundary_right(x: np.ndarray, on_boundary: bool) -> bool:
        """Returns True if the input point is on the right-hand boundary, 
        and False otherwise.
        """
        return on_boundary and (x[0] > 1.0 - 1e-8)
    
    nx, ny = 32, 32

    mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(1.0, 1.0), nx, ny)
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

    f = dl.Constant(0.0)

    u0_left = dl.Expression("1.0 + 0.5*x[1]", degree=1)
    u0_right = dl.Expression("-sin(2.0*pi*x[1]) - 1.0", degree=1)
    bcs = [
        dl.DirichletBC(Vh, u0_left, boundary_left), 
        dl.DirichletBC(Vh, u0_right, boundary_right)
    ]

    coords = torch.from_numpy(Vh.tabulate_dof_coordinates())
    dirichlet_mask = torch.bitwise_or(
        abs(coords[:, 0]) < 1e-8, 
        abs(coords[:, 0] - 1.0) < 1e-8
    )

    x0_obs = torch.linspace(0.2, 0.8, 4)
    x1_obs = torch.linspace(0.2, 0.8, 4)
    xs_obs = torch.tensor([[x0, x1] for x0 in x0_obs for x1 in x1_obs])

    model = PoissonSolver(mesh, Vh, f, bcs, dirichlet_mask, xs_obs)

    mu_pri = 0.0
    std_pri = 1.0 
    ls_pri = 1.0 / math.sqrt(50.0)
    kernel = SquaredExponential(std_pri, ls_pri)
    prior = GaussianRandomField(coords, mu_pri, kernel, num_kl=num_kl)  # type: ignore

    if not rom:
        return prior, model

    # Specify parameters for constructing ROM
    num_snapshots = 1000
    eps = 1.0e-3

    xs = prior.sample(n=num_snapshots)
    log_ks = prior.transform(xs)
    
    us_snap = torch.vstack([
        model.solve(log_ks_i) 
        for log_ks_i in log_ks
    ]).T

    pod_basis = compute_pod_basis(us_snap, eps)
    model_rom = PoissonSolverROM(model, pod_basis)
    # print(f"Reduced basis size: {pod_basis.shape[1]}.")

    return prior, model_rom