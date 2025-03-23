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
from dolfinx.mesh import create_unit_square, locate_entities
from dolfinx.plot import vtk_mesh
from ufl import (
    Measure, 
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
    def side_boundary_indicator(x: np.ndarray) -> np.ndarray[bool]:
        """Returns a boolean which indicates whether a point is on the 
        left- or right-hand side of the domain.
        """
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)

    @staticmethod
    def side_boundaries(x: np.ndarray) -> np.ndarray:
        """Ensures that the value of the solution is equal to 1 on the
        left-hand boundary and 0 on the right-hand boundary.
        """
        return 1.0 - x[0]
    
    def kappa(self, xs: np.ndarray, ks: np.ndarray) -> np.ndarray[bool]:
        """Returns the value of the diffusion coefficient at a set of 
        points in the domain.

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

# Locate cell indices within domain (function omega_0) returns booleans that indicate whether points are in the domain
# cells_0 = locate_entities(solver.mesh, solver.mesh.topology.dim, Omega_0)
# dx_equivalent = Measure("dx", domain=solver.mesh, subdomain_id=cells_0)

import dolfinx

def inner_square(x):
    return x[0] < 0.5

tdim = solver.mesh.topology.dim
cell_map = solver.mesh.topology.index_map(tdim)
num_cells = cell_map.size_local + cell_map.num_ghosts
all_cells = np.arange(num_cells, dtype=np.int32)

marker = np.ones(num_cells, dtype=np.int32)

t0 = time.time()
marker[dolfinx.mesh.locate_entities(solver.mesh, tdim, inner_square)] = 2
cell_tag = dolfinx.mesh.meshtags(solver.mesh, tdim, np.arange(num_cells, dtype=np.int32), marker)

dx_alt = Measure("dx", domain=solver.mesh, subdomain_data=cell_tag)

# dx_alt = Measure("dx", domain=solver.mesh)
area = uh * dx_alt
compiled_area = dolfinx.fem.form(area)
local_area = dolfinx.fem.assemble_scalar(compiled_area)
global_area = solver.mesh.comm.allreduce(local_area, op=MPI.SUM)
# print(global_area)

t1 = time.time()
print(t1-t0)

m = 9  # From Cui and Dolgov...

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