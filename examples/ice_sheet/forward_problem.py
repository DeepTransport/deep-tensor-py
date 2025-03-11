import copy
import os
import sys

import dolfin
from dolfin import *
import numpy as np
from ufl import nabla_div
import matplotlib.pyplot as plt

import time

import torch 
from torch import Tensor

import deep_tensor as dt

plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)


class PeriodicBoundary(SubDomain):

    def __init__(self, length: float):
        self.length = length
        dolfin.cpp.mesh.SubDomain.__init__(self)
        return
    
    def inside(self, x: np.ndarray, on_boundary: bool):
        return np.isclose(x[0], 0.0) and on_boundary
    
    def map(self, x, y):
        y[0] = x[0] - self.length
        y[1] = x[1]
        return


class Top(SubDomain):

    def __init__(self, height: float):
        self.height = height 
        dolfin.cpp.mesh.SubDomain.__init__(self)
        return

    def inside(self, x, on_boundary):
        return near(x[1], self.height)


class Bottom(SubDomain):

    def inside(self, x, on_boundary):
        return near(x[1], 0)


class IceSheetModel():

    def __init__(
        self, 
        length: float = 10_000, 
        height: float = 1_000, 
        nx: int = 50, 
        ny: int = 5
    ):

        self.nx = nx 
        self.ny = ny

        # Define length and height of domain (m^2), angle of domain 
        # (radians), and gravitational acceleration (m/s^2)
        self.length = length
        self.height = height 
        self.angle = 0.1 * torch.pi / 180.0
        self.grav = 9.81

        # Define ice density
        self.rho = 910.0

        # Define [unknown] constants and Rheology parameter
        self.sConst = Constant(1.0e-08)
        self.scale_const = Constant(1.0e-10)
        self.N = 3.0

        # Define ice flow parameter and pre-flow prefactor
        self.Aconst = Constant(2.140373 * 1.0e+07)
        self.Aconst_NL = 1.0 / self.Aconst if self.N == 1 else 10.0 ** -16.0

        # Define boundary conditions
        self.top = Top(height=self.height)
        self.sides = PeriodicBoundary(length=self.length)
        self.bottom = Bottom()

        # Define mesh
        x0 = Point(0, 0)
        x1 = Point(self.length, self.height)
        self.mesh = RectangleMesh(x0, x1, self.nx, self.ny)
        self.boundary_mesh = BoundaryMesh(self.mesh, "exterior", True)

        # Define function spaces
        # x and y, 2=order (need higher order for velocity)
        self.P2 = VectorElement(
            family="Lagrange", 
            cell=self.mesh.ufl_cell(), 
            degree=2
        )
        self.P1 = FiniteElement(
            family="Lagrange", 
            cell=self.mesh.ufl_cell(), 
            degree=1
        )
        self.TH = self.P2 * self.P1

        # Define periodic product spaces for velocity and pressure
        self.VQ = FunctionSpace(self.mesh, self.TH, constrained_domain=self.sides) 
        self.VP = FunctionSpace(self.mesh, "Lagrange", 1)
        self.Vh = [self.VQ, self.VP, self.VQ]

        # Define forcing function in x and y directions
        self.f = Expression( 
            cpp_code=("rho * grav * sin(angle)", "-rho * grav * cos(angle)"),
            element=self.VQ.sub(0).ufl_element(), 
            angle=self.angle, 
            grav=self.grav, 
            rho=self.rho
        )

        self.bc = DirichletBC(self.VQ.sub(0).sub(1), Constant(0.0), self.bottom)

        boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)

        self.Gamma_B = Bottom()
        self.Gamma_B.mark(boundary_markers, 1)

        self.ds = Measure(
            integral_type="ds", 
            domain=self.mesh, 
            subdomain_data=boundary_markers
        )

        self.normal = FacetNormal(self.mesh)
        self.n = interpolate(Constant(float(self.N)), self.VP) 
        return

    @staticmethod
    def Tang(u, normal): 
        """Tangential operator (PetraEtAl2012, Eq. (3))."""
        return (u - outer(normal, normal) * u)

    @staticmethod
    def epsilon(u):
        """Strains."""
        return Constant(0.5) * (nabla_grad(u) + nabla_grad(u).T)

    def solve(
        self, 
        beta: str, 
        linesearch: bool = True, 
        rtol: float = 1e-08,
        atol: float = 1e-08,
        max_iter: int = 100,
        max_back_it: int = 50,
        c_armijo: float = 1e-04,
        verbose: bool = False
    ):
        
        # Define energy functional and its first and second variations
        def Energy(u):
            normEu12 = 0.5 * inner(self.epsilon(u), self.epsilon(u)) + self.sConst
            phi = ((2.0 * self.n) / (self.n + 1.0)) * self.Aconst_NL**(-1.0 / self.n) * normEu12 ** ((self.n + 1.0) / (2.0 * self.n))
            E = (phi * dx - inner(self.f, u) * dx + Constant(0.5) * inner(exp(beta) * self.Tang(u, self.normal), self.Tang(u, self.normal)) * self.ds(1))
            return self.scale_const * E

        def Gradient(u, v, p, q):  # u, test func for u, pressure, test func for pressure
            """Computes the gradient...

            Parameters
            ----------
            u:
                Velocity.
            v:
                Test functions for velocity (adjoint velocity).
            p:
                Pressure.
            q: 
                Test functions for pressure.

            """
            normEu12 = 0.5 * inner(self.epsilon(u), self.epsilon(u)) + self.sConst

            return self.scale_const * (self.Aconst_NL ** (-1.0/n) * ((normEu12**((1.0 - n)/(2.0 * n))) * inner(self.epsilon(u), self.epsilon(v))) * dx - \
                inner(self.f, v) * dx + inner(exp(beta) * self.Tang(u, self.normal), self.Tang(v, self.normal)) * self.ds(1) - \
                p * nabla_div(v) * dx - q * nabla_div(u) * dx)

        # Next variations
        # p=pressure, q=test func for pressure, r=direction you are applying it in.
        def Hessian(u, v, w, p, q, r):
            normEu12 = 0.5*inner(self.epsilon(u), self.epsilon(u)) + self.sConst
            
            return self.scale_const * (self.Aconst_NL ** (-1.0 / n)
                                * (((1.0-n)/(2.0*n))*(normEu12**((1.0-3.0*n) / (2.0*n)) * \
                inner(self.epsilon(u), self.epsilon(w)) * inner(self.epsilon(v), self.epsilon(u))) + \
                ((normEu12**((1.0 - n) / (2.0 * n))) * inner(self.epsilon(w), self.epsilon(v)))) * dx + \
                inner(exp(beta)*self.Tang(w, self.normal), self.Tang(v, self.normal)) * ds(1) - \
                r*nabla_div(v)*dx - q*nabla_div(w)*dx)

        # Solve linear problem for initial guess
        uh, ph = TrialFunctions(self.VQ)
        vh, qh = TestFunctions(self.VQ)

        # Linear version of Stokes
        self.a_linear = (self.Aconst * inner(self.epsilon(uh), self.epsilon(vh)) * dx 
                         - div(vh) * ph * dx 
                         - qh * div(uh) * dx 
                         + inner(exp(beta) * self.Tang(uh, self.normal), self.Tang(vh, self.normal)) * self.ds(1))
        self.L_linear = inner(self.f, vh) * dx

        vq = Function(self.VQ)

        old_stdout = sys.stdout # backup current stdout
        sys.stdout = open(os.devnull, "w")
        solve(self.a_linear == self.L_linear, vq, self.bc)  # initial guess for nonlinear solve... 
        sys.stdout = old_stdout

        uh_lin, ph_lin = vq.split(deepcopy=True)
        self.uh = uh_lin

        n = interpolate(Constant(self.N), self.VP) 

        u, p = vq.split()
        v, q = TestFunctions(self.VQ)
        w, r = TrialFunctions(self.VQ)

        dvq = Function(self.VQ)
        dvq.assign(vq)

        J = Energy(u)
        G = Gradient(u, v, p, q)
        H = Hessian(u, v, w, p, q, r)

        Ju0 = assemble(J)
        Jn = copy.deepcopy(Ju0)     # For line search
        g0 = assemble(G)            # Initial gradient
        g0_norm = g0.norm("l2")

        for i in range(max_iter):
            
            Hn, gn = assemble_system(H, G, self.bc)
            Hn.init_vector(dvq.vector(), 1)

            solve(Hn, dvq.vector(), gn)
            gn_norm = gn.norm("l2")
            dvq_gn = -dvq.vector().inner(gn)

            if verbose: 
                print(f"Iteration {i+1}...")

            # Initialise stepsize
            alpha = 1.0

            if linesearch:

                vq_back = vq.copy(deepcopy=True)
                bk_converged = False
                
                for _ in range(max_back_it):

                    vq.vector().axpy(-alpha, dvq.vector())
                    u, p = vq.split()

                    J = Energy(u)
                    Jnext = assemble(J)
                    
                    if Jnext < Jn + abs(alpha * c_armijo * dvq_gn):
                        Jn = Jnext
                        bk_converged = True
                        break
                    
                    alpha = alpha / 2.0
                    vq.assign(vq_back)
                
                if not bk_converged:
                    vq.assign(vq_back)
                    if verbose:
                        print("Max backtracking iterations completed.")
                    break
            else:
                vq.vector().axpy(-alpha, dvq.vector())
                u, p = vq.split()
                J = Energy(u)
                Jn = assemble(J)

            if verbose:    
                print(gn_norm)
            G = Gradient(u, v, p, q)
            H = Hessian(u, v, w, p, q, r)

        u, p = vq.split(deepcopy=True)
        # ufile_pvd = File("nonlinear_velocity.pvd")
        # ufile_pvd << u

        return u, p

    def get_observations(self, u) -> Tensor:
        """Gets some observations from the velocity field.
        """

        u_vals = u.compute_vertex_values(self.mesh).reshape(2, -1).T
        u_vals = torch.tensor(u_vals)

        ux = u_vals[:, 0].reshape(self.ny+1, self.nx+1)
        uy = u_vals[:, 1].reshape(self.ny+1, self.nx+1)

        ux_top = ux[-1]
        uy_top = uy[-1]

        # TODO: move elsewhere
        obs_inds = torch.tensor([10, 20, 30, 40], dtype=torch.int)
        
        d = torch.hstack((ux_top[obs_inds], uy_top[obs_inds]))
        return d

    def plot_velocities(self, u, fname="velocities"):

        u_vals = u.compute_vertex_values(self.mesh).reshape(2, -1).T
        coords = self.mesh.coordinates()
        coords = np.array(coords)

        x_vals = np.linspace(0, self.length, self.nx+1)
        y_vals = np.linspace(0, self.height, self.ny+1)

        ux_vals = u_vals[:, 0].reshape(self.ny+1, self.nx+1)
        uy_vals = u_vals[:, 1].reshape(self.ny+1, self.nx+1)

        fig, axes = plt.subplots(2, 1, figsize=(10, 2.8), sharex=True)

        axes[0].set_aspect("equal", adjustable="box")
        axes[1].set_aspect("equal", adjustable="box")

        u0_mesh = axes[0].pcolormesh(x_vals, y_vals, ux_vals)
        u1_mesh = axes[1].pcolormesh(x_vals, y_vals, uy_vals)

        fig.colorbar(u0_mesh, ax=axes[0], label=r"$u_{1}$ [m\,a$^{-1}$]")
        fig.colorbar(u1_mesh, ax=axes[1], label=r"$u_{2}$ [m\,a$^{-1}$]")

        axes[1].set_xlabel(r"$x_{1}$ [m]")
        axes[0].set_ylabel(r"$x_{2}$ [m]")
        axes[1].set_ylabel(r"$x_{2}$ [m]")

        plt.savefig(f"examples/ice_sheet/{fname}.pdf")
        return
    
    def plot_beta(self, beta, fname="beta"):

        beta_vals = beta.compute_vertex_values(model.mesh)
        coords = model.mesh.coordinates()
        coords = np.array(coords)

        x_vals = np.linspace(0, model.length, model.nx+1)
        beta_vals = beta_vals.reshape(model.ny+1, model.nx+1)[0]

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(x_vals, beta_vals)
        ax.set_xlabel(r"$x_{1}$ [m]")
        ax.set_ylabel(r"$\beta$")

        plt.savefig(f"examples/ice_sheet/{fname}.pdf")
        return


class BetaFunc(UserExpression):
    
    def __init__(
        self, 
        mean: Tensor,
        field: np.ndarray, 
        nx: int, 
        length: float, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mean = mean
        self.field = field
        self.nx = nx
        self.length = length
        return
    
    def eval(self, value, x):
        i = (self.nx * x[0]) // self.length
        value[:] = self.mean + self.field[int(i)]
        return
    
    def value_shape(self):
        return ()


class BetaPrior():
    """A prior on the basal sliding coefficient."""
    
    def __init__(
        self, 
        mean: float|Tensor, 
        cov: Tensor,
        n_modes: int = 8
    ):
        
        self.mean = mean
        self.n_modes = n_modes

        eigvals, eigvecs = torch.linalg.eigh(cov)
        idx = eigvals.argsort(descending=True)[:self.n_modes]
        
        self.eigvals = eigvals[idx]
        self.eigvecs = eigvecs[:, idx] * self.eigvals.sqrt()
        
        return
    
    def get_beta(self, xi: Tensor):  # TODO: add model in here.
        """Converts a set of KL coefficients to the correponding 
        basal sliding coefficient.
        """

        field = self.eigvecs @ xi

        rf = BetaFunc(self.mean, field, model.nx, model.length, degree=1)
        beta = Function(model.VP)
        beta.assign(interpolate(rf, beta.function_space()))
        
        return beta


model = IceSheetModel()

# Generate prior
mean = 7.0
xs = torch.linspace(0.0, model.length, model.nx+1)
dxs = torch.tensor([[torch.linalg.norm(x0-x1) for x0 in xs] for x1 in xs])
cov = 2.0 ** 2 * torch.exp(-0.5 * dxs ** 2 / 100.0**2)

prior = BetaPrior(mean, cov)

xis = torch.normal(mean=0.0, std=1.0, size=(prior.n_modes,))

beta = prior.get_beta(xis)

# t0 = time.time()
u, p = model.solve(beta)
# t1 = time.time()
# print(t1-t0)

model.plot_velocities(u)
model.plot_beta(beta)

# Specify likelihood
std_d = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1])
cov_d = torch.diag(std_d.square())
L_d = torch.linalg.cholesky(torch.linalg.inv(cov_d))

# Generate some synthetic data
d = model.get_observations(u)
d_obs = d + torch.normal(mean=0.0, std=std_d)

def potential_func(xs: Tensor):

    n_xs = xs.shape[0]
    print(n_xs)

    potential_pri = xs.square().sum(dim=1)
    potential_lik = torch.zeros(n_xs)

    for i in range(n_xs):
        t0 = time.time()
        beta = prior.get_beta(xs[i])
        u = model.solve(beta)[0]
        fx_i = model.get_observations(u)
        potential_lik[i] = (L_d @ (fx_i - d_obs)).square().sum()
        t1 = time.time()
        # print(t1-t0)

    print(potential_pri+potential_lik)

    return potential_pri + potential_lik


bounds = torch.tensor([-4.0, 4.0])
domain = dt.BoundedDomain(bounds=bounds)
poly = dt.Lagrange1(num_elems=30)
bases = dt.ApproxBases(polys=poly, domains=domain, dim=prior.n_modes)

options = dt.TTOptions(max_cross=1, init_rank=10)

sirt = dt.TTSIRT(potential_func, bases=bases, options=options)