from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv
import torch 

from docs.examples.hgg.grfs import MaternField3D

xs = torch.linspace(0.0, 100.0, 100)
ys = torch.linspace(0.0, 100.0, 100)

nx, ny, nz = 20, 30, 40

xrng = np.linspace(-5, 5, nx)
yrng = np.linspace(-8, 8, ny)
zrng = np.linspace(-7, 4, nz)
points = np.array([[x, y, z] for z in zrng for y in yrng for x in xrng])
x, y, z = np.meshgrid(xrng, yrng, zrng, indexing='ij')
# x = x.swapaxes(0, 2)
# y = y.swapaxes(0, 2)
# z = z.swapaxes(0, 2)
grid = pv.StructuredGrid(x, y, z)# .triangulate()

# mesh = pv.StructuredGrid(xs.numpy(), y=ys.numpy())

lx, ly, lz = 10.0, 10.0, 10.0
ls = np.array([lx, ly, lz]) 

from pathlib import Path
folder = Path(__file__).parent.joinpath("data", "test_grf").resolve()

field = MaternField3D(grid, ls, folder=folder)

W = torch.randn(field.num_points)
sigma = 1.0
x = field.generate_field(W, sigma)
field.plot(x)
x = x.reshape(nx, ny, nz)

# plt.pcolormesh(x[:, :, 0])
# plt.show()