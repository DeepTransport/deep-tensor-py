"""Example 3.3 from Cui and Dolgov, 2022."""

import time

from dolfinx import default_scalar_type
from dolfinx.fem import (
    Constant, 
    Function, 
    dirichletbc, 
    functionspace, 
    locate_dofs_geometrical
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square
from dolfinx.plot import vtk_mesh
from ufl import (
    TestFunction, 
    TrialFunction, 
    dot, 
    dx, 
    grad
)
from mpi4py import MPI
import numpy as np
import pyvista as pv


class DiffusionSolver():

    def __init__(
        self,
        nx: int = 64, 
        ny: int = 64,
        dim_k: int = 11,
        nu: int = 2
    ):
        """Initialises diffusion equation solver.

        Parameters
        ----------
        nx: 
            Number of discretisation points in the x direction.
        ny: 
            Number of discretisation points in the y direction.
        dim_k: 
            Dimension of parametrisation for diffusion coefficient.
        nu: 
            Component of parametrisation for diffusion coefficient.
            
        """
        
        # Define mesh
        self.mesh = create_unit_square(comm=MPI.COMM_WORLD, nx=nx, ny=ny)

        # Define space composed of piecewise linear basis functions
        self.V = functionspace(mesh=self.mesh, element=("Lagrange", 1))

        # Define trial and test functions
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        # Define Dirichlet boundary condition
        dofs_D = locate_dofs_geometrical(self.V, self.side_boundary_indicator)
        u_bc = Function(self.V)
        u_bc.interpolate(self.side_boundaries)
        self.dirichlet = dirichletbc(u_bc, dofs_D)

        # Generate parameters associated with the diffusion coefficient
        self.dim_k = dim_k
        ks = np.arange(1, dim_k+1)
        sum_ks = np.sum(ks ** -(nu + 1.0))
        taus = np.floor(0.5 * (np.sqrt(1.0 + 0.5*ks) - 1.0))

        self.etas = (ks ** -(nu + 1.0)) / sum_ks
        self.rho_0s = ks - 0.5 * (taus**2 + taus)
        self.rho_1s = taus - self.rho_0s
        return

    @staticmethod
    def side_boundary_indicator(x: np.ndarray) -> bool:
        """Returns a boolean which indicates whether a point is on the 
        left- or right-hand side of the domain.
        """
        return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))

    @staticmethod
    def side_boundaries(x: np.ndarray) -> float:
        """Ensures that the value of the solution is equal to 1 on the
        left-hand boundary and 0 on the right-hand boundary.
        """
        return 1.0 - x[0]
    
    def kappa(self, xs: np.ndarray, ks: np.ndarray) -> float:
        """Returns the value of the diffusion coefficient at a given 
        point in the domain.

        Parameters
        ----------
        xs: 
            A matrix containing the points of the domain. Each column 
            contains a single (three-dimensional) point.
        ks:
            The random coefficients associated with the parametrisation 
            of kappa.
        
        Returns
        -------
        k:
            The value of the diffusion coefficient.
        
        """
        x0 = xs[0][:, None]
        x1 = xs[1][:, None]
        log_k = np.sum(ks * np.sqrt(self.etas) 
                       * np.cos(2.0 * np.pi * self.rho_0s * x0) 
                       * np.cos(2.0 * np.pi * self.rho_1s * x1), axis=1)
        k = np.exp(log_k)
        return k
    
    def solve(self, ks: np.ndarray) -> Function:

        def _kappa(xs: np.ndarray) -> float:
            return self.kappa(xs, ks)

        self.k = Function(self.V)
        self.k.interpolate(_kappa)

        # Define left- and right-hand sides of variational formulation
        a = self.k * dot(grad(self.u), grad(self.v)) * dx
        L = Constant(self.mesh, default_scalar_type(0)) * self.v * dx

        problem = LinearProblem(a=a, L=L, bcs=[self.dirichlet])
        uh = problem.solve()
        return uh

solver = DiffusionSolver()

ks = 2.0 * np.sqrt(3.0) * np.random.rand(solver.dim_k) - np.sqrt(3.0)

t0 = time.time()
uh = solver.solve(ks)
t1 = time.time()
print(t1-t0)

# V2 = functionspace(mesh, ("Lagrange", 2))
# uex = Function(V2)
# uex.interpolate(u_exact)
# error_L2 = assemble_scalar(form((uh - uex)**2 * dx))
# error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(error_L2, op=MPI.SUM))

# u_vertex_values = uh.x.array
# uex_1 = Function(V)
# uex_1.interpolate(uex)
# u_ex_vertex_values = uex_1.x.array
# error_max = np.max(np.abs(u_vertex_values - u_ex_vertex_values))
# error_max = MPI.COMM_WORLD.allreduce(error_max, op=MPI.MAX)
# print(f"Error_L2 : {error_L2:.2e}")
# print(f"Error_max : {error_max:.2e}")

pv.start_xvfb()

pyvista_cells, cell_types, geometry = vtk_mesh(solver.V)
grid = pv.UnstructuredGrid(pyvista_cells, cell_types, geometry)
grid.point_data["u"] = uh.x.array
grid.set_active_scalars("u")

plotter = pv.Plotter()
plotter.add_text("u_h", position="upper_edge", font_size=14, color="black")
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()

plotter.show()