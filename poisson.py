import time

from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import torch 
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor


GRAD_2D = torch.tensor([[-1.0, 1.0, 0.0], 
                        [-1.0, 0.0, 1.0]])

def on_dirichlet_boundary(x: Tensor) -> bool:
    return x[0].isclose(torch.tensor(0.0)) | x[0].isclose(torch.tensor(1.0))

def dirichlet_bc(x: Tensor) -> Tensor:
    """0 on LHS, 1 on RHS."""
    return x[0]

def get_squared_distances(points: Tensor) -> Tensor:
    n, d = points.shape
    x = points.unsqueeze(1).expand(n, n, d)
    y = points.unsqueeze(0).expand(n, n, d)
    d_sq = (x-y).square().sum(dim=2)
    return d_sq


class PoissonProblem():

    def __init__(
        self, 
        nx: int = 64,
        ny: int = 64
    ):
        
        self.nx = nx 
        self.ny = ny
        
        xs = torch.linspace(0.0, 1.0, self.nx)
        ys = torch.linspace(0.0, 1.0, self.ny)

        self.dx = xs[1] - xs[0]
        self.dy = ys[1] - ys[0]
        self.jac = self.dx * self.dy

        self.points = torch.tensor([[x, y] for x in xs for y in ys])
        self.elements = self._build_elements(nx, ny)

        self._build_fem_matrices(nx, ny, self.points, self.elements)
        return
    
    def _build_elements(self, nx: int, ny: int) -> Tensor:

        # Form tensor containing all elements
        elements = []
        for i in range((nx-1) * ny):
            if ((i+1) % ny) != 0:
                elements.extend([[i, i+1, i+ny+1], [i, i+ny, i+ny+1]])

        return elements
    
    def _build_fem_matrices(self, nx, ny, points, elements) -> None:

        K_i, K_j, K_v = [], [], []
        D_i, D_j, D_v = [], [], []
        f = torch.zeros(nx*ny)

        n = 0

        # Fill in all non-boundary elements
        for e in elements:

            for i in range(3):

                T = torch.vstack([points[e[(i+1)%3]] - points[e[i]],
                                  points[e[(i+2)%3]] - points[e[i]]]).T
                
                detT = torch.abs(torch.linalg.det(T))
                invT = torch.linalg.inv(T)

                if not on_dirichlet_boundary(points[e[i]]):

                    for j in range(3):

                        k = 1/2 * detT * GRAD_2D[:, 0] @ invT @ invT.T @ GRAD_2D[:, (j-i)%3]

                        K_i.append(e[i])
                        K_j.append(e[j])
                        K_v.append(k)

                        D_i.extend([n, n, n])
                        D_j.extend(e)
                        D_v.extend([1.0/3.0] * 3)

                        n += 1

        # Fill in boundary elements
        for i, point in enumerate(points):
            if on_dirichlet_boundary(point):
                K_i.append(i)
                K_j.append(i)
                K_v.append(1.0)
                f[i] = dirichlet_bc(point)

        self.K = sparse.coo_matrix((K_v, (K_i, K_j)), shape=(nx*ny, nx*ny))
        self.D = sparse.coo_matrix((D_v, (D_i, D_j)), shape=(n, nx*ny))
        self.f = f        
        self.n = n
        return

    def solve(self, log_ks: Tensor) -> Tensor:

        K_solve = self.K.copy()
        K_solve.data[:self.n] *= self.D @ log_ks.exp()
        K_solve = K_solve.tocsr()

        us = spsolve(K_solve, self.f)
        return us
    
prob = PoissonProblem()

ell = 0.20
d_sq = get_squared_distances(prob.points)
cov = torch.exp(-d_sq/(2.0*ell**2)) + 1e-4 * torch.eye(prob.nx * prob.ny)

prior = MultivariateNormal(loc=torch.zeros((prob.nx * prob.ny,)), covariance_matrix=cov)
log_ks = prior.sample((1000, ))

for log_ks_i in log_ks:
    t0 = time.time()
    u = prob.solve(log_ks_i)
    t1 = time.time()
    print(t1-t0)

# plt.pcolormesh(u.reshape(prob.ny, prob.nx))
# plt.colorbar()
# plt.show()

# plt.pcolormesh(log_ks.reshape(prob.ny, prob.nx))
# plt.colorbar()
# plt.show()