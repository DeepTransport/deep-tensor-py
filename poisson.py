import torch 
from torch import Tensor


GRAD_2D = torch.tensor([[-1.0, 1.0, 0.0], 
                        [-1.0, 0.0, 1.0]])

def on_dirichlet_boundary(x: Tensor) -> bool:
    return x[0].isclose(torch.tensor(0.0)) | x[0].isclose(torch.tensor(1.0))

def dirichlet_bc(x: Tensor) -> Tensor:
    """0 on LHS, 1 on RHS."""
    return x[0]


# TODO: Add variable diffusion coefficient


# Number of nodes in each direction
nx = 64
ny = 64

xs = torch.linspace(0.0, 1.0, nx)
ys = torch.linspace(0.0, 1.0, ny)

dx = xs[1] - xs[0]
dy = ys[1] - ys[0]
jac = dx * dy

points = torch.tensor([[x, y] for x in xs for y in ys])

n = 0

# Form tensor containing all elements
elements = []
for i in range((nx-1)*ny):
    if ((i+1) % ny) != 0:
        elements.extend([[i, i+1, i+ny+1], 
                         [i, i+ny, i+ny+1]])

elements = torch.tensor(elements)
print(elements[:10])

K_i = []
K_j = []
K_v = []

for e in elements:

    for i in range(3):

        T = torch.vstack([points[e[(i+1)%3]] - points[e[i]],
                          points[e[(i+2)%3]] - points[e[i]]]).T
        
        detT = torch.abs(torch.linalg.det(T))
        invT = torch.linalg.inv(T)

        for j in range(3):

            k = 1/2 * detT * GRAD_2D[:, 0] @ invT @ invT.T @ GRAD_2D[:, (j-i)%3]

            K_i.append(e[i])
            K_j.append(e[j])
            K_v.append(k)

K_inds = torch.tensor((K_i, K_j))
vals = torch.tensor(K_v)

from scipy import sparse

K = sparse.coo_matrix((K_v, (K_i, K_j)), shape=(nx*ny, nx*ny)).toarray()

f = torch.zeros(nx*ny)

for i, point in enumerate(points):
    if on_dirichlet_boundary(point):
        K[i, :] = 0.0
        K[i, i] = 1.0
        f[i] = dirichlet_bc(point)

import time 
t0 = time.time()
K = sparse.csr_matrix(K)
p = sparse.linalg.spsolve(K, f)
t1 = time.time()
print(t1-t0)
p = torch.tensor(p)
p = p.reshape(ny, nx)

from matplotlib import pyplot as plt
plt.pcolormesh(p)
plt.colorbar()
plt.show()


# TODO: make a list of all the triangles in the mesh (include the top and bottom triangles in each square)... need indices of points
# Loop through each triangle and compute 