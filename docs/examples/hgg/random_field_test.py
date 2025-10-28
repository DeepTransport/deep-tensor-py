from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv
import torch 

from docs.examples.hgg.grfs import MaternField3D

xs = torch.linspace(0.0, 100.0, 100)
ys = torch.linspace(0.0, 100.0, 100)

nx, ny, nz = 100, 100, 25

xrng = np.linspace(-5, 5, nx)
yrng = np.linspace(-8, 8, ny)
zrng = np.linspace(-7, 4, nz)
x, y, z = np.meshgrid(xrng, yrng, zrng, indexing='xy')
grid = pv.StructuredGrid(x, y, z).triangulate()


# mesh = pv.StructuredGrid(xs.numpy(), y=ys.numpy())

field = MaternField3D(grid)  # type: ignore

W = torch.randn(field.num_points)
sigma = 1.0
lx, ly, lz = 2.0, 2.0, 2.0
x = field.generate_field(W, sigma, lx, ly, lz)
field.plot(x)
x = x.reshape(nx, ny, nz)

# plt.pcolormesh(x[:, :, 0])
# plt.show()