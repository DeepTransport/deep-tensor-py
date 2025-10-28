from matplotlib import pyplot as plt 
import torch

# Note: currently assuming equal spacing between points in each 
# dimension.
dx = 1.0
dy = 1.0

nx = 50
ny = 50
n = nx * ny

Lx = torch.zeros((n, n))
Ly = torch.zeros((n, n))

for i in range(n):

    on_boundary_0 = i % ny == 0
    on_boundary_1 = i % ny == ny - 1

    if on_boundary_0:
        # Lx[i, i+2] = 1.0 / (dx**2)
        # Lx[i, i+1] = -2.0 / (dx**2) 
        # Lx[i, i] = 1.0 / (dx**2)
        Lx[i, i+1] = -1.0 / dx
        Lx[i, i] = 1.0 / dx
    elif on_boundary_1:
        # Lx[i, i] = 1.0 / (dx**2) 
        # Lx[i, i-1] = -2.0 / (dx**2)
        # Lx[i, i-2] = 1.0 / (dx**2)
        Lx[i, i] = 1.0 / dx
        Lx[i, i-1] = -1.0 / dx
    else:
        Lx[i, i] = -2.0 / (dx**2)
        Lx[i, i+1] = 1.0 / (dx**2)
        Lx[i, i-1] = 1.0 / (dx**2)

for i in range(n):

    on_boundary_0 = i < ny
    on_boundary_1 = i >= ny * (nx-1)

    if on_boundary_0:
        Ly[i, i+ny] = 1.0 / dy
        Ly[i, i] = -1.0 / dy
    elif on_boundary_1:
        Ly[i, i] = 1.0 / dy
        Ly[i, i-ny] = -1.0 / dy
    else:
        Ly[i, i] = -2.0 / (dy**2)
        Ly[i, i+ny] = 1.0 / (dy**2)
        Ly[i, i-ny] = 1.0 / (dy**2)


L = Lx + Ly 

p_sqrt = -L + 100 * torch.eye(n)

rs = torch.randn((n))

xs = p_sqrt @ rs 
for i in range(5):
    xs = p_sqrt @ xs 

xs = xs.reshape(nx, ny)
plt.pcolormesh(xs)
plt.colorbar()
plt.show()