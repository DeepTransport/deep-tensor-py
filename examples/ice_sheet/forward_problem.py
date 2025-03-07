import copy

import dolfin
from dolfin import *
import numpy as np
from ufl import nabla_div
import matplotlib.pyplot as plt

import time

plt.style.use("examples/plotstyle.mplstyle")
# np.random.seed(1)


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
        height: float = 3_000, 
        nx: int = 150, 
        ny: int = 45
    ):

        self.nx = nx 
        self.ny = ny

        # Define length and height of domain (m^2), angle of domain 
        # (radians), and gravitational acceleration (m/s^2)
        self.length = length
        self.height = height 
        self.angle = 0.1 * np.pi / 810.0
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
        self.Gamma_B.mark(boundary_markers,1)

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
        """Tangential operator."""
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
        verbose: bool = True
    ):
        
        def Energy(u):
            normEu12 = 0.5 * inner(self.epsilon(u), self.epsilon(u)) + self.sConst
            return self.scale_const * (self.Aconst_NL**(-1.0 / self.n)*((2.0 * self.n) / (1.0 + self.n)) * (normEu12 ** ((1.0 + self.n) / (2.0 * self.n))) * dx - \
                inner(self.f, u) * dx + Constant(0.5) * inner(exp(beta) * self.Tang(u, self.normal), \
                self.Tang(u, self.normal)) * self.ds(1))

        def Gradient(u, v, p, q):  # u, test func for u, pressure, test func for pressure
            normEu12 = 0.5*inner(self.epsilon(u), self.epsilon(u)) + self.sConst
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
        solve(self.a_linear == self.L_linear, vq, self.bc)  # initial guess for nonlinear solve... 

        uh_lin, ph_lin = vq.split(deepcopy=True)
        self.uh = uh_lin

        # 5 Define exponential parameter field and Energy functional, and 1st and 2nd variations

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
                vq.vector().axpy(-alpha,dvq.vector())
                u, p = vq.split()
                J = Energy(u)
                Jn = assemble(J)
                
            print(gn_norm)
            G = Gradient(u, v, p, q)
            H = Hessian(u, v, w, p, q, r)

        u, p = vq.split(deepcopy=True)
        ufile_pvd = File("nonlinear_velocity.pvd")
        ufile_pvd << u

        return beta, u, p

model = IceSheetModel()

# beta_code = "std::log(1001.0 + 1000.0 * sin(x[0] * 2.0 * pi / length))"
# beta = Expression(
#     cpp_code=beta_code,
#     element=model.VP.ufl_element(), 
#     length=model.length
# )

class BetaRF(UserExpression):
    
    def __init__(self, field: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.field = field
        return
    
    def eval(self, value, x):
        i = (model.nx * x[0]) // model.length
        value[:] = 5.0 + self.field[int(i)]
        return
    
    def value_shape(self):
        return ()

xs = np.linspace(0.0, model.length, model.nx+1)
dxs = np.array([[np.linalg.norm(x0-x1) for x0 in xs] for x1 in xs])
cov = 2.0 ** 2 * np.exp(-0.5 * dxs ** 2 / 100.0**2)

n_modes = 8

eigvals, eigvecs = np.linalg.eigh(cov)
idx = eigvals.argsort()[::-1][:n_modes]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
eigvecs = eigvecs * np.sqrt(eigvals)

xis = np.random.normal(size=(n_modes, ))

field = eigvecs @ xis

rf = BetaRF(field, degree=1)
beta = Function(model.VP)
beta.assign(interpolate(rf, beta.function_space()))

t0 = time.time()
beta, u, p = model.solve(beta)
t1 = time.time()

print(t1-t0)

# u_vec = u.vector().get_local()
# print(u_vec)

beta_vals = beta.compute_vertex_values(model.mesh)
u_vals = u.compute_vertex_values(model.mesh).reshape(2, -1).T
coords = model.mesh.coordinates()
coords = np.array(coords)

x_vals = np.linspace(0, model.length, model.nx+1)
y_vals = np.linspace(0, model.height, model.ny+1)

u_vals_x = u_vals[:, 0].reshape(model.ny+1, model.nx+1)
u_vals_y = u_vals[:, 1].reshape(model.ny+1, model.nx+1)
beta_vals = beta_vals.reshape(model.ny+1, model.nx+1)[0]

fig, axes = plt.subplots(3, 1, figsize=(7, 6))#, sharex=True)

axes[0].set_aspect("equal", adjustable="box")
axes[1].set_aspect("equal", adjustable="box")

u0_mesh = axes[0].pcolormesh(x_vals, y_vals, u_vals_x)
u1_mesh = axes[1].pcolormesh(x_vals, y_vals, u_vals_y)
axes[2].plot(x_vals, beta_vals)

# fig.colorbar(u0_mesh, ax=axes[0])
# fig.colorbar(u1_mesh, ax=axes[1])

axes[0].set_title(r"$u_{1}$")
axes[1].set_title(r"$u_{2}$")

axes[2].set_xlabel(r"$x_{1}$ [m]")
axes[0].set_ylabel(r"$x_{2}$ [m]")
axes[1].set_ylabel(r"$x_{2}$ [m]")
axes[2].set_ylabel(r"$\beta$")

plt.savefig("examples/ice_sheet/coef_and_velocities.pdf")