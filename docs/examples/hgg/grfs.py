from enum import Enum

import numpy as np
import pyvista as pv
from scipy import sparse
from scipy.special import gamma
from sksparse.cholmod import cholesky
import tqdm


GRAD_2D = np.array([[-1.0, 1.0, 0.0], 
                    [-1.0, 0.0, 1.0]])

GRAD_3D = np.array([[-1.0, 1.0, 0.0, 0.0], 
                    [-1.0, 0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0, 1.0]])

class BC(Enum):
    NEUMANN = 1
    ROBIN = 2


# Code for triangular elements in PyVista
TRIANGLE = 10

import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        t0 = time.perf_counter()
        value = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"Finished in {(t1-t0):.1f} seconds.")
        return value
    return wrapper_timer


class MaternField3D():

    def __init__(
        self, 
        mesh: pv.StructuredGrid, 
        ls: float | np.ndarray,
        bc_type: BC = BC.ROBIN, 
        lam: float | None = None, 
        folder: str=""
    ):
        
        self.dim = 3
        self.nu = 2 - self.dim / 2
        if isinstance(ls, float) or isinstance(ls, int):
            ls = np.array([ls, ls, ls])
        self.ls = ls
        self.ls_prod = self.ls[0] * self.ls[1] * self.ls[2]
        self.bc_type = bc_type
        if lam is None:
            # TODO: see what FENiCS does.
            lam = 1.0e+3 * self.ls_prod ** (1.0/3.0)
        self.lam = lam

        self.mesh = mesh.triangulate()
        self.folder = folder
        self.get_mesh_data()
        self.load_fem_matrices()
        return
        
    def get_mesh_data(self):
        """Extracts information on the points, elements and facets of the 
        mesh.
        """

        self.num_points = self.mesh.n_points 
        self.num_elements = self.mesh.n_cells

        self.mesh["inds"] = np.arange(self.num_points)  # type: ignore

        self.points = self.mesh.points
        self.elements = self.mesh.cells_dict[TRIANGLE]

        boundary = self.mesh.extract_geometry()
        boundary_points = boundary.cast_to_pointset()["inds"]
        boundary_facets = boundary.faces.reshape(-1, 4)[:, 1:]
        self.boundary_facets = [boundary_points[f] for f in boundary_facets]

        self.num_boundary_facets = len(self.boundary_facets)
        return

    def load_fem_matrices(self):
        
        try:
            self.M = sparse.load_npz(f"{self.folder}/M.npz")
            self.Kx = sparse.load_npz(f"{self.folder}/Kx.npz")
            self.Ky = sparse.load_npz(f"{self.folder}/Ky.npz")
            self.Kz = sparse.load_npz(f"{self.folder}/Kz.npz")
            self.N = sparse.load_npz(f"{self.folder}/N.npz")
            self.L = sparse.load_npz(f"{self.folder}/L.npy")
        except FileNotFoundError:
            print("FEM matrices not found. Constructing...")
            self.build_fem_matrices()
            # sparse.save_npz(f"{self.folder}/M", self.M)
            # sparse.save_npz(f"{self.folder}/Kx", self.Kx)
            # sparse.save_npz(f"{self.folder}/Ky", self.Ky)
            # sparse.save_npz(f"{self.folder}/Kz", self.Kz)
            # sparse.save_npz(f"{self.folder}/N", self.N)
            # sparse.save_npz(f"{self.folder}/L", self.L)
        return

    def build_fem_matrices(self):
        """Builds the FEM matrices required to generate Matern fields 
        in three dimensions.
        """

        M_i = np.zeros((16 * self.num_elements,))
        M_j = np.zeros((16 * self.num_elements,))
        M_v = np.zeros((16 * self.num_elements,))

        K_i = np.zeros((16 * self.num_elements,))
        K_j = np.zeros((16 * self.num_elements,))
        K_v = np.zeros((3, 16 * self.num_elements))

        N_i = np.zeros((9 * self.num_boundary_facets,))
        N_j = np.zeros((9 * self.num_boundary_facets,))
        N_v = np.zeros((9 * self.num_boundary_facets,))

        n = 0
        for e in tqdm.tqdm(self.elements):

            for i in range(4):

                T = np.array([self.points[e[(i+1)%4]] - self.points[e[i]],
                              self.points[e[(i+2)%4]] - self.points[e[i]],
                              self.points[e[(i+3)%4]] - self.points[e[i]]]).T

                detT = np.abs(np.linalg.det(T))
                invT = np.linalg.inv(T)

                for j in range(4):
                    
                    M_i[n] = e[i]
                    M_j[n] = e[j]
                    M_v[n] = (detT * 1/60) if i == j else (detT * 1/120)

                    kl = 1/6 * detT * GRAD_3D[:, 0].T @ invT
                    kr = invT.T @ GRAD_3D[:, (j-i)%4]

                    K_i[n] = e[i]
                    K_j[n] = e[j]
                    K_v[:, n] = kl.flatten() * kr.flatten()

                    n += 1
        
        n = 0
        for f in self.boundary_facets:

            for i in range(3):
                
                detTf = np.linalg.norm(np.cross(self.points[f[(i+1)%3]] - self.points[f[i]], 
                                                self.points[f[(i+2)%3]] - self.points[f[i]]))

                for j in range(3):
                    
                    N_i[n] = f[i]
                    N_j[n] = f[j]
                    N_v[n] = (detTf * 1/12) if i == j else (detTf * 1/24)

                    n += 1

        shape = (self.num_points, self.num_points)

        self.M = sparse.coo_matrix((M_v, (M_i, M_j)), shape=shape)
        self.Kx = sparse.coo_matrix((K_v[0], (K_i, K_j)), shape=shape)
        self.Ky = sparse.coo_matrix((K_v[1], (K_i, K_j)), shape=shape)
        self.Kz = sparse.coo_matrix((K_v[2], (K_i, K_j)), shape=shape)
        self.N = sparse.coo_matrix((N_v, (N_i, N_j)), shape=shape)

        K = (self.ls[0] ** 2 * self.Kx 
             + self.ls[1] ** 2 * self.Ky 
             + self.ls[2] ** 2 * self.Kz)
        
        self.A = self.M + K
        if self.bc_type == BC.ROBIN:
            self.A += (self.ls_prod / self.lam) * self.N 

        self.chol = cholesky(self.A)
        
        mass_lumps = np.sqrt(self.M.sum(axis=1)).flatten()
        self.L = sparse.spdiags(mass_lumps, diags=0, m=self.num_points, n=self.num_points)

        print("FEM matrices constructed.")
        return

    @timer
    def generate_field(self, W, sigma):
        """Generates a Matern field."""

        alpha = sigma**2 * (2**self.dim * np.pi**(self.dim/2) * \
                            gamma(self.nu + self.dim/2)) / gamma(self.nu)

        b = (alpha * self.ls_prod) ** 0.5 * self.L @ W

        # x1 = sparse.linalg.spsolve(self.A, b)

        x = self.chol.solve_A(b)

        # y = sparse.linalg.spsolve_triangular(self.chol, b, lower=True)
        # x = sparse.linalg.spsolve_triangular(self.chol.T, y, lower=False)
        return x
    
    def plot(self, values, **kwargs):
        p = pv.Plotter()
        p.add_mesh(self.mesh, scalars=values, **kwargs) # type: ignore
        p.show()
        return