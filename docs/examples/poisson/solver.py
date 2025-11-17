import dolfin as dl
import hippylib as hl
import numpy as np
import torch
from torch import Tensor

from examples.priors import GaussianRandomField

dl.set_log_level(dl.LogLevel.WARNING)


def build_obs_operator(Vh: dl.FunctionSpace, xs: Tensor) -> Tensor:
    B = hl.assemblePointwiseObservation(Vh, xs)
    B = torch.from_numpy(B.array())
    return B


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


class PoissonSolver(object):
    
    def __init__(
        self, 
        mesh, 
        V: dl.FunctionSpace, 
        prior: GaussianRandomField, 
        xs_obs: Tensor
    ):
        
        self.mesh = mesh
        self.V = V
        self.prior = prior

        self.xs_obs = xs_obs
        self.num_obs = self.xs_obs.shape[0]
        self.B = build_obs_operator(self.V, self.xs_obs)

        self.u = dl.TrialFunction(self.V)
        self.v = dl.TestFunction(self.V)

        u0_left = dl.Expression("1.0 + 0.5*x[1]", degree=1)
        u0_right = dl.Expression("-sin(2.0*pi*x[1]) - 1.0", degree=1)
        self.bc_left = dl.DirichletBC(V, u0_left, boundary_left)
        self.bc_right = dl.DirichletBC(V, u0_right, boundary_right)
        
        self.f = dl.Constant(0.0)
        self.rhs = self.f * self.v * dl.dx
        return

    def vec2func(self, vec: dl.Vector | Tensor) -> dl.Function:
        """Converts a dl.Vector or torch.Tensor to a dl.Function."""

        if isinstance(vec, dl.Vector):
            vec = vec.get_local()[:]

        if isinstance(vec, Tensor):
            vec = vec.numpy()

        k = dl.Function(self.V)
        k.vector().set_local(vec)
        k.vector().apply("insert")
        return k

    def solve(self, k: Tensor) -> Tensor:
        """Solves the forward problem for a given coefficient field."""

        k = self.vec2func(k)
        K = dl.exp(k) * dl.inner(dl.grad(self.u), dl.grad(self.v)) * dl.dx

        u = dl.Function(self.V)
        dl.solve(K == self.rhs, u, bcs=[self.bc_left, self.bc_right])
        return torch.from_numpy(u.vector()[:])
    
    def observe(self, us: Tensor) -> Tensor:
        return self.B @ us
    
    def eval_jacobian(self, log_ks: Tensor):

        num_nodes = log_ks.numel()

        log_ks = self.vec2func(log_ks)
        varf = dl.exp(log_ks) * dl.inner(dl.grad(self.u), dl.grad(self.v)) * dl.dx  # type: ignore

        # Compute mask containing nodes on Dirichlet boundary
        coords = self.V.tabulate_dof_coordinates()
        dirichlet_nodes = np.bitwise_or(abs(coords[:, 0]) < 1e-8, abs(coords[:, 0] - 1.0) < 1e-8)

        # Form A matrix and forcing term
        A = dl.assemble(varf)
        f = dl.assemble(self.rhs)
        self.bc_left.apply(A, f)
        self.bc_right.apply(A, f)

        # Define LU solver for A matrix (to be re-used later)
        LU_solver = dl.LUSolver(A)

        # Solve incremental forward problem
        u_inc = dl.Vector()
        A.init_vector(u_inc, 0)
        LU_solver.solve(u_inc, f)

        # Define basis functions for derivative of k
        z = dl.interpolate(dl.Constant(0.0), self.V)
        
        col = dl.Vector()
        A.init_vector(col, 0)

        dAdku_inc = torch.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            
            # Create a function that is equal to 1 at the current node,
            # and 0 elsewhere
            z_c = z.copy(deepcopy=True)  # type: ignore
            z_c.vector()[i] = 1.0

            # Assemble variational form for current node
            varf_grad = dl.assemble(
                z_c * dl.exp(log_ks) * dl.inner(dl.grad(self.u), dl.grad(self.v)) * dl.dx  # type: ignore
            )

            # Take product of assembled matrix and incremental solution 
            # to obtain column i of the derivative
            varf_grad.mult(u_inc, col)
            dAdku_inc[:, i] = torch.tensor(col[:])

        # Deal with Dirichlet nodes
        dAdku_inc[dirichlet_nodes, :] = 0.0

        prod = torch.zeros((num_nodes, num_nodes))

        LU_solver = dl.LUSolver(A)

        u = dl.Vector()
        A.init_vector(u, 0)

        rhs = dl.Vector()
        A.init_vector(rhs, 0)
        
        for i in range(num_nodes):
            rhs.set_local(dAdku_inc[:, i].numpy())
            LU_solver.solve(u, rhs)
            prod[:, i] = torch.tensor(u[:])

        jac = -self.B @ prod
        return jac