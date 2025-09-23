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

        # TODO: need to define u0 (i.e., the values on the Dirichlet 
        # boundaries) to align with @CuiEtAl2025.
        # TODO: check degree argument.
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