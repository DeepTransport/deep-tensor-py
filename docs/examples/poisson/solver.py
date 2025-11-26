from typing import List

import dolfin as dl
import hippylib as hl
import numpy as np
import torch
from torch import Tensor

from examples.priors import GaussianRandomField

from reduced_order_modelling import compute_pod_basis

dl.set_log_level(dl.LogLevel.WARNING)


# TODO: fix this. PyOED has a Python implementation which forms the 
# observation operator which could provide some inspiration
def build_obs_operator(Vh: dl.FunctionSpace, xs: Tensor) -> Tensor:
    B = hl.assemblePointwiseObservation(Vh, xs)
    B = torch.from_numpy(B.array())
    return B


def vec2func(vec: dl.Vector | Tensor, Vh: dl.FunctionSpace) -> dl.Function:
    """Converts a dl.Vector or torch.Tensor to a dl.Function."""

    if isinstance(vec, dl.Vector):
        vec = vec.get_local()[:]  # type: ignore

    if isinstance(vec, Tensor):
        vec = vec.numpy()

    k = dl.Function(Vh)
    k.vector().set_local(vec)
    k.vector().apply("insert")
    return k


def func2vec(func: dl.Function | dl.Vector) -> Tensor:
    return torch.from_numpy(func.vector()[:])


def build_channel(xs: Tensor) -> Tensor:
    """Builds something similar to the channel diffusivity 
    parametrisation from CuiEtAl2024.
    """
    x, y = xs.T
    in_channel = torch.bitwise_and(
        y >= 0.07 * torch.sin(8.0*x) - 0.2*x + 0.45,
        y <= 0.07 * torch.sin(8.0*x) - 0.2*x + 0.60
    )
    k = 0.01 + 100.0 * in_channel
    return k.log()


class PoissonSolver(object):
    """Solves the Poisson equation.
    
    Parameters
    ----------
    mesh:
        Mesh.
    Vh:
        FEM function space.
    f:
        Forcing function.
    bcs: 
        List of Dirichlet boundary conditions.
    dirichlet_mask:
        Boolean mask which indicates which nodes are part of the 
        Dirichlet boundary.
    prior: 
        Prior.
    xs_obs:
        A two-dimensional tensor, where each row contains the spatial 
        location of an observation.
    
    """
    
    def __init__(
        self, 
        mesh, 
        Vh: dl.FunctionSpace, 
        f: dl.Constant | dl.Function,
        bcs: List[dl.DirichletBC],
        dirichlet_mask: Tensor,
        xs_obs: Tensor
    ):
        
        self.mesh = mesh
        self.Vh = Vh
        self.bcs = bcs
        self.dirichlet_mask = dirichlet_mask

        self.xs_obs = xs_obs
        self.num_obs = self.xs_obs.shape[0]
        self.B = build_obs_operator(self.Vh, self.xs_obs)

        self.u = dl.TrialFunction(self.Vh)
        self.v = dl.TestFunction(self.Vh)
        self.f = dl.assemble(f * self.v * dl.dx)  # type: ignore
        self.apply_bcs(self.f)

        return

    def apply_bcs(self, A) -> None:
        for bc in self.bcs:
            bc.apply(A)
        return

    def solve(self, log_k: Tensor) -> Tensor:
        """Solves the forward problem for a given coefficient field."""

        # Assemble stiffness matrix
        log_k = vec2func(log_k, self.Vh)  # type: ignore
        A = dl.exp(log_k) * dl.inner(dl.grad(self.u), dl.grad(self.v)) * dl.dx  # type: ignore
        A = dl.assemble(A)
        self.apply_bcs(A)

        u = dl.Vector()
        A.init_vector(u, 0)
        dl.solve(A, u, self.f, "lu")

        return torch.from_numpy(u[:])
    
    def observe(self, us: Tensor) -> Tensor:
        return self.B @ us
    
    def eval_jacobian(self, log_k: Tensor) -> Tensor:
        """Evaluates the Jacobian of the parameter-to-observable 
        mapping for a given value of the unknown coefficient.
        """

        num_nodes = log_k.numel()

        log_k = vec2func(log_k, self.Vh)  # type: ignore
        A = dl.exp(log_k) * dl.inner(dl.grad(self.u), dl.grad(self.v)) * dl.dx  # type: ignore
        A = dl.assemble(A)
        self.apply_bcs(A)

        # Define LU solver for A matrix (to be re-used later)
        LU_solver = dl.LUSolver(A)

        # Solve incremental forward problem
        u_inc = dl.Vector()
        A.init_vector(u_inc, 0)
        LU_solver.solve(u_inc, self.f)

        # Define basis functions for derivative of k
        z = dl.interpolate(dl.Constant(0.0), self.Vh)

        dAdku_inc = torch.zeros((num_nodes, num_nodes))
        col = dl.Vector()
        A.init_vector(col, 0)
        
        for i in range(num_nodes):

            z_c = z.copy(deepcopy=True)  # type: ignore
            z_c.vector()[i] = 1.0

            # Assemble variational form for current node
            varf_grad = z_c * dl.exp(log_k) * dl.inner(dl.grad(self.u), dl.grad(self.v)) * dl.dx  # type: ignore
            varf_grad = dl.assemble(varf_grad)

            # Take product of assembled matrix and incremental solution 
            # to obtain column i of the derivative
            varf_grad.mult(u_inc, col)
            dAdku_inc[:, i] = torch.tensor(col[:])

        # Deal with Dirichlet nodes
        dAdku_inc[self.dirichlet_mask] = 0.0

        prod = torch.zeros((num_nodes, num_nodes))

        LU_solver = dl.LUSolver(A)

        u = dl.Vector()
        A.init_vector(u, 0)

        rhs = dl.Vector()
        A.init_vector(rhs, 0)
        
        # Note: ideally, we would compute BA^-1 (which requires num_obs
        # linear solves), but I haven't figured out how to solve a 
        # linear system with the transpose of A yet.
        for i in range(num_nodes):
            rhs.set_local(dAdku_inc[:, i].numpy())
            LU_solver.solve(u, rhs)
            prod[:, i] = torch.tensor(u[:])

        jac = -self.B @ prod
        return jac


class PoissonSolverROM():
    
    def __init__(self, model_full: PoissonSolver, basis: Tensor):

        self.model_full = model_full
        self.V = basis
        self.dim_x, self.dim_r = self.V.shape
        
        self.Vh = self.model_full.Vh
        self.dirichlet_mask = self.model_full.dirichlet_mask
        self.num_obs = self.model_full.num_obs
        self.B = self.model_full.B

        self._build_rom_matrices()
        return

    def _build_rom_matrices(self) -> None:
        """Builds and reduces the set of matrices multiplied by the 
        nodal values of the log-diffusion coefficient to form the full 
        A matrix, after making a piecewise linear approximation to the 
        log-diffusion coefficient.
        """

        u = self.model_full.u 
        v = self.model_full.v
        
        # Compute all the submatrices that combine to form A
        z = dl.interpolate(dl.Constant(0.0), self.Vh)

        A_rs = torch.zeros((self.dim_r, self.dim_r, self.dim_x))

        for i in range(self.dim_x):
            
            z_c = z.copy(deepcopy=True)  # type: ignore
            z_c.vector()[i] = 1.0
            A = dl.assemble(z_c * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx)  # type: ignore
            A = torch.tensor(A.array())

            # Zero out all Dirichlet rows (these are accounted for in 
            # the final matrix)
            A[self.dirichlet_mask, :] = 0.0
            A_rs[:, :, i] = self.V.T @ A @ self.V

        # Form section for Dirichlet BCs
        A_bcs = torch.zeros((self.dim_x, self.dim_x))
        A_bcs[self.dirichlet_mask, self.dirichlet_mask] = 1.0 
        self.A_r_bcs = self.V.T @ A_bcs @ self.V

        self.f_r = self.V.T @ torch.tensor(self.model_full.f[:])
        self.A_rs = A_rs
        return
    
    def reduced_to_full(self, u_r: Tensor) -> Tensor:
        return self.V @ u_r

    def solve_full(self, log_k: Tensor) -> Tensor:
        """Computes the pressure field associated with a given 
        log-diffusivity field using the full model.
        """
        return self.model_full.solve(log_k)

    def solve(self, log_k: Tensor) -> Tensor:
        """Computes the pressure field associated with a given 
        log-diffusivity field using the reduced-order model.
        """
        A_r = torch.einsum("ijk, k", self.A_rs, log_k.exp()) + self.A_r_bcs
        u_r = torch.linalg.solve(A_r, self.f_r)
        u = self.reduced_to_full(u_r)
        return u
    
    def observe(self, us: Tensor) -> Tensor:
        return self.model_full.observe(us)
    
    def eval_jacobian(self, log_k: Tensor) -> Tensor:

        # Compute (reduced) A matrix
        A_r = torch.einsum("ijk, k", self.A_rs, log_k.exp()) + self.A_r_bcs

        # Solve (reduced) incremental forward problem
        u_inc = torch.linalg.solve(A_r, self.f_r)

        # Compute product of dAdk and solution of incremental adjoint problem
        dAdk_u = torch.einsum("ilj, l", self.A_rs * log_k.exp()[None, None, :], u_inc)

        # Solve incremental adjoint problems
        adj = torch.linalg.solve(A_r.T, self.V.T @ self.B.T).T
        jac = -adj @ dAdk_u
        return jac


if __name__ == "__main__":

    import math
    import time

    from matplotlib import pyplot as plt

    from docs.examples.priors import SquaredExponential
    from docs.examples.plotting import plot_dl_function

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)


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

    x0_obs = torch.linspace(0.1, 0.9, 4)
    x1_obs = torch.linspace(0.1, 0.9, 4)
    xs_obs = torch.tensor([[x0, x1] for x0 in x0_obs for x1 in x1_obs])

    model = PoissonSolver(mesh, Vh, f, bcs, dirichlet_mask, xs_obs)

    mu_pri = 0.0
    std_pri = 1.0 
    ls_pri = 1.0 / math.sqrt(50.0)
    kernel = SquaredExponential(std_pri, ls_pri)
    prior = GaussianRandomField(coords, mu_pri, kernel)  # type: ignore

    RUN_FORWARD = False 
    CHECK_JACOBIAN = False 
    BUILD_ROM = False
    CHECK_ROM_JACOBIAN = False
    RUN_CONTAMINANT = True

    if RUN_FORWARD:

        log_k = prior.transform(prior.sample(n=1).flatten())
        u = model.solve(log_k)

        u = vec2func(u, model.Vh)

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_dl_function(fig, ax, vec2func(log_k, Vh))
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_dl_function(fig, ax, vec2func(u, Vh))
        plt.show()

    if CHECK_JACOBIAN:

        log_k = prior.transform(prior.sample(n=1).flatten())

        # Compute analytical Jacobian
        jac = model.eval_jacobian(log_k)

        # Approximate single column using finite differences
        col_ind = 161
        dx = 1.0e-7
        log_ks_0 = log_k.clone()
        log_ks_1 = log_k.clone()
        log_ks_0[col_ind] -= dx
        log_ks_1[col_ind] += dx
        u_0 = model.observe(model.solve(log_ks_0))
        u_1 = model.observe(model.solve(log_ks_1))
        jac_fd = (u_1-u_0)/(2*dx)

        errors = jac[:, col_ind] / jac_fd
        print(jac[:, col_ind])
        print(jac_fd)
        print(errors)
        print(f"Max error (%): {100*(errors-1.0).abs().max()}")

    if BUILD_ROM:

        num_snapshots = 1000
        eps = 1.0e-3
        num_validation_solves = 1000

        xs = prior.sample(n=num_snapshots)
        log_ks = prior.transform(xs)
        
        us_snap = torch.vstack([
            model.solve(log_ks_i) 
            for log_ks_i in log_ks
        ]).T

        pod_basis = compute_pod_basis(us_snap, eps)
        print(f"Reduced basis size: {pod_basis.shape[1]}.")

        model_rom = PoissonSolverROM(model, pod_basis)

        log_ks = prior.transform(prior.sample(n=num_validation_solves))

        t0 = time.time()
        us = torch.vstack([model.solve(log_k) for log_k in log_ks])
        t1 = time.time()
        us_rom = torch.vstack([model_rom.solve(log_k) for log_k in log_ks])
        t2 = time.time()

        print(f"Full model mean solve time: {(t1-t0)/num_validation_solves:.2e} s.")
        print(f"ROM mean solve time: {(t2-t1)/num_validation_solves:.2e} s.")

        stds = us.std(dim=0)
        means_error = (us-us_rom).mean(dim=0)
        stds_error = (us-us_rom).std(dim=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_dl_function(fig, ax, vec2func(stds, Vh))
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_dl_function(fig, ax, vec2func(means_error, Vh))
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_dl_function(fig, ax, vec2func(stds_error, Vh))
        plt.show()

        if CHECK_ROM_JACOBIAN:
            
            jac = model_rom.eval_jacobian(log_ks[0])

            col_ind = 3
            dx = 1.0e-7
            log_ks_0 = log_ks[0].clone()
            log_ks_1 = log_ks[0].clone()
            log_ks_0[col_ind] -= dx
            log_ks_1[col_ind] += dx
            u_0 = model_rom.observe(model_rom.solve(log_ks_0))
            u_1 = model_rom.observe(model_rom.solve(log_ks_1))
            jac_fd = (u_1-u_0)/(2*dx)

            print(jac[:, col_ind])
            print(jac_fd)

            errors = jac[:, col_ind] / jac_fd
            print(jac[:, col_ind])
            print(jac_fd)
            print(errors)
            print(f"Max error (%): {100*(errors-1.0).abs().max()}")

    if RUN_CONTAMINANT:

        from solver_contaminant import ContaminantSolver

        log_ks = prior.transform(prior.sample(n=100))
        log_ks[0] = build_channel(coords)

        t0 = time.time()
        us = torch.vstack([model.solve(log_k) for log_k in log_ks])
        t1 = time.time()
        print(t1-t0)

        xs = torch.linspace(0.0, 1.0, nx+1)
        ys = torch.linspace(0.0, 1.0, ny+1)

        contaminant_solver = ContaminantSolver(xs, ys, coords)

        t_breaks = contaminant_solver.solve(log_ks, us)
        print(t_breaks[1:].min())
        t2 = time.time()
        print(t2-t1)
        print(f"Breakthrough time: {t_breaks[0]:.4f} s.")

        # plt.quiver(xs, ys, -kdudx[:, :, 0], -kdudx[:, :, 1])

        for i in range(us.shape[0]):
            path = contaminant_solver.solve_path(log_ks[i], us[i])
            flux = -dl.project(dl.exp(vec2func(log_ks[i], model.Vh)) * dl.grad(vec2func(us[i], model.Vh)))  # type: ignore
            dl.plot(flux)
            plt.scatter([0], [0.5])
            plt.plot(*path.T)
            plt.show()