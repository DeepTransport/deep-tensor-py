import dolfin as dl
import hippylib as hl
import numpy as np
import torch
from torch import Tensor
import ufl

from ..priors import ProcessConvolutionPrior


torch.set_default_dtype(torch.float64)


VariableContainer = list[hl.TimeDependentVector, dl.Vector, hl.TimeDependentVector]


def get_state_matrix(u: hl.TimeDependentVector) -> Tensor:
    return torch.hstack([torch.tensor(u_i[:])[:, None] for u_i in u.data])


class HeatSolver(object):
    
    def __init__(
        self, 
        mesh, 
        V: dl.FunctionSpace, 
        prior: ProcessConvolutionPrior, 
        ts: Tensor,
        xs_obs: Tensor,
        obs_ind_0: int,
        obs_dx: int,
        B: Tensor
    ):
        
        self.mesh = mesh
        self.V = V
        self.prior = prior

        self.dt = ts[1] - ts[0]  # NOTE: assumes equispaced times
        self.nt = len(ts)
        self.ts = ts
        self.xs_obs = xs_obs
        self.obs_ind_0 = obs_ind_0 
        self.obs_dx = obs_dx
        self.B = B

        self.u = dl.TrialFunction(self.V)
        self.v = dl.TestFunction(self.V)
        self.u0 = dl.interpolate(dl.Constant(0.0), self.V).vector()
        
        # Define forcing function
        sd = "-1/(2*std::pow(r, 2))"
        f0 = "(std::pow(x[0]-a0, 2) + std::pow(x[1]-a1, 2))"
        f1 = "(std::pow(x[0]-b0, 2) + std::pow(x[1]-b1, 2))"
        self.f = dl.Expression(
            f"c * (std::exp({sd} * {f0}) - std::exp({sd} * {f1}))",
            c=5.0*np.pi*1e-2,
            r=0.05,
            a0=0.5, a1=0.5,
            b0=2.5, b1=0.5,
            element=V.ufl_element()
        )
        self.f = dl.interpolate(self.f, self.V).vector()

        # Assemble mass matrix
        self.M = dl.assemble(self.u * self.v * ufl.dx)
        return

    def vec2func(self, vec: dl.Vector | Tensor) -> dl.Function:
        """Converts a vector to a function."""

        if isinstance(vec, dl.Vector):
            vec = vec.get_local()[:]

        if isinstance(vec, Tensor):
            vec = vec.numpy()

        k = dl.Function(self.V)
        k.vector().set_local(vec)
        k.vector().apply("insert")
        return k

    def sample_prior(self) -> dl.Vector:
        """Returns a single vector sampled from the prior.
        """
        # Generate vector of white noise
        noise = dl.Vector()
        self.prior.init_vector(x=noise, dim="noise")
        hl.parRandom.normal(sigma=1.0, out=noise)
        # Generate prior sample
        ks = dl.Vector()
        self.prior.init_vector(x=ks, dim=0)
        self.prior.sample(noise=noise, s=ks)
        return ks

    def generate_vector(
        self, 
        component: str | int = "ALL"
    ) -> VariableContainer | hl.TimeDependentVector | dl.Vector:
        """Generates an empty vector (or set of vectors) of the 
        appropriate dimension.
        """

        if component == "ALL":
            u = hl.TimeDependentVector(self.ts)
            u.initialize(self.M, 0)
            m = dl.Vector(dl.MPI.comm_world, self.prior.dim)
            p = hl.TimeDependentVector(self.ts)
            p.initialize(self.M, 0)
            return [u, m, p]
        
        elif component == hl.STATE:
            u = hl.TimeDependentVector(self.ts)
            u.initialize(M=self.M, dim=0)
            return u
        
        elif component == hl.PARAMETER:
            m = dl.Vector()
            self.prior.init_vector(x=m, dim=0)
            return m
        
        elif component == hl.ADJOINT:
            p = hl.TimeDependentVector(self.ts)
            p.initialize(M=self.M, dim=0)
            return p
        
        raise Exception(f"Unknown component: '{component}'.")
    
    def init_parameter(self, m: dl.Vector) -> None:
        """Initialises a parameter vector such that it has the 
        appropriate dimension.
        """
        self.prior.init_vector(m, 0)
        return

    def solve(self, k: Tensor) -> Tensor:
        """Solves the forward problem.

        Time is discretised using the backward Euler method.
        
        Parameters
        ----------
        TODO
        
        """

        us = torch.zeros((self.mesh.num_vertices(), self.nt))

        # Initialise state vector and right-hand side
        u = dl.Vector()
        self.M.init_vector(u, 0)
        rhs = dl.Vector()
        self.M.init_vector(rhs, 0)

        # Initialise previous state
        u_prev = self.u0.copy()

        # Assemble stiffness matrix
        k = self.vec2func(k)
        K = ufl.exp(k) * ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        K = dl.assemble(K)
                
        # Define LHS of variational form
        A = self.M + self.dt * K
        solver = dl.LUSolver(A)
        
        for i in range(self.nt):
            self.M.mult(self.dt * self.f + u_prev, rhs)
            solver.solve(u, rhs)
            us[:, i] = torch.tensor(u[:])
            u_prev = u.copy()
        
        return us
    
    def observe(self, us: Tensor) -> Tensor:
        d_obs = (self.B @ us)[:, self.obs_ind_0::self.obs_dx]
        return d_obs.T.flatten()

class HeatSolverROM(object):

    def __init__(
        self, 
        solver: HeatSolver, 
        m: Tensor, 
        V: Tensor, 
        K_rs: Tensor
    ):

        self.solver = solver 
        self.mesh = solver.mesh
        self.prior = solver.prior
        self.m = m
        self.V = V
        self.r = V.shape[1]

        self.u = solver.u 
        self.v = solver.v

        self.ts = solver.ts
        self.dt = solver.dt
        self.nt = solver.nt
        self.M_r = V.T @ torch.tensor(solver.M.array()) @ V
        self.K_rs = K_rs
        self.f_r = V.T @ torch.tensor(solver.f[:])
        self.u0_r = V.T @ torch.tensor(solver.u0[:])

        return

    def reduced_to_full(self, u: Tensor) -> Tensor:
        return self.V @ u

    def generate_vector(
        self, 
        component: str | int = "ALL"
    ) -> VariableContainer | hl.TimeDependentVector | dl.Vector:
        return self.solver.generate_vector(component)

    def solve(self, k: Tensor) -> Tensor:
        """Solves the forward problem in the reduced space.

        Time is discretised using the backward Euler method.
        
        Parameters
        ----------
        out:
            A set of state vectors at each time snapshot. Each vector 
            is stored in out.data.
        x:
            A list of the state, parameter and adjoint vectors.
        
        """

        # Initialise previous state
        u_r_prev = self.u0_r.clone()

        # Assemble reduced stiffness matrix
        K_r = torch.einsum("ijk, k", self.K_rs, torch.exp(k))

        # Define LHS of variational form
        A_r = self.M_r + self.dt * K_r
        lu = torch.linalg.lu_factor(A_r)

        us = torch.zeros((self.mesh.num_vertices(), self.nt))

        for i in range(self.nt):

            b_r = self.M_r @ (self.dt * self.f_r + u_r_prev)

            u_r: Tensor = torch.linalg.lu_solve(*lu, b_r[:, None]).flatten()

            us[:, i] = self.reduced_to_full(u_r)
            u_r_prev = u_r.clone()

        return us
    
    def observe(self, us: Tensor) -> Tensor:
        return self.solver.observe(us)