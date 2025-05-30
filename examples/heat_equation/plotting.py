import dolfin as dl
from matplotlib import pyplot as plt, tri
import numpy as np


plt.style.use("examples/plotstyle.mplstyle")


def triangulate(coords, cells) -> tri.Triangulation:
    return tri.Triangulation(coords[:, 0], coords[:, 1], cells)


def plot_function(
    func: dl.Function, 
    **kwargs
): 

    mesh = func.function_space().mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()

    triangulation = triangulate(coords, cells)
    vals = func.compute_vertex_values(mesh)

    dx = np.max(coords[:, 0]) - np.min(coords[:, 0])
    dy = np.max(coords[:, 1]) - np.min(coords[:, 1])

    fig, ax = plt.subplots(figsize=(dx/dy*2.0, 2.0))

    col = ax.tripcolor(triangulation, vals, **kwargs)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    fig.colorbar(col, ax=ax)
    return