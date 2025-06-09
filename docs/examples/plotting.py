import dolfin as dl
from matplotlib import tri
import numpy as np


def triangulate(coords, cells) -> tri.Triangulation:
    return tri.Triangulation(coords[:, 0], coords[:, 1], cells)


def plot_dl_function(
    fig, ax,
    func: dl.Function, 
    cbar_label: str | None = None,
    **kwargs
) -> None: 

    mesh = func.function_space().mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()

    triangulation = triangulate(coords, cells)
    vals = func.compute_vertex_values(mesh)

    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)

    col = ax.tripcolor(triangulation, vals, **kwargs)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    cbar = fig.colorbar(col, ax=ax)
    cbar.set_label(cbar_label)
    return


def add_arrows(ax) -> None:
    """Removes the top spine and right spine from a plot, and adds 
    arrows to the ends of the bottom spine and left spine.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(1, 0, ">k", markersize=6, transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", markersize=6, transform=ax.transAxes, clip_on=False)
    return