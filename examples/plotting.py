"""TODO: move into main package one day?"""

from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import torch 
from torch import Tensor

import deep_tensor as dt


def corner_plot(
    irt: dt.TTSIRT|dt.TTDIRT, 
    samples: Tensor,
    fname: str,
    labels: list[str]|None = None
) -> None:
    """Generates a corner plot for the joint distribution associated 
    with an SIRT object.
    """

    if labels is None:
        # Use default parameter labels
        labels = [f"$x_{i+1}$" for i in range(irt.dim)]

    _, axes = plt.subplots(
        irt.dim, 
        irt.dim, 
        figsize=(2*irt.dim, 2*irt.dim),
        sharex=True
    )

    samples_min = samples.min(dim=0).values
    samples_max = samples.max(dim=0).values
    samples_range = samples_max - samples_min

    lims_min = samples_min - 0.05 * samples_range
    lims_max = samples_max + 0.05 * samples_range
    
    bounds = torch.hstack((lims_min[:, None], lims_max[:, None]))

    for ax in axes.flat:
        ax.set_box_aspect(1)

    for i in range(irt.dim):
        for j in range(i+1):

            if i == j:
                
                # Plot marginal
                kde = gaussian_kde(samples[:, j])
                xs = torch.linspace(*bounds[j], 100)
                density = kde(xs)

                axes[i][j].plot(xs, density, c="k", lw=1.5)
                axes[i][j].yaxis.set_ticklabels([])
                axes[i][j].set_ylim(0.0, 1.1 * max(density))
            
            else:
                
                # Plot joint
                kde = gaussian_kde(samples[:, [j, i]].T)
                xs = torch.linspace(*bounds[j], 100)
                ys = torch.linspace(*bounds[i], 100)
                zs = torch.tensor([[x, y] for x in xs for y in ys])
                kde_zs = kde(zs.T).reshape(100, 100).T

                axes[i][j].set_xlim(*bounds[j])
                axes[i][j].set_ylim(*bounds[i])
                axes[i][j].contour(xs, ys, kde_zs, levels=12, linewidths=1.5)
                axes[i][j].yaxis.set_ticklabels([])
            
            if i == irt.dim-1:
                axes[i][j].tick_params(axis="y", width=0.0)
                axes[i][j].tick_params(axis="x", direction="in", width=0.5)
                axes[i][j].set_xlabel(labels[j]) 
            else:
                axes[i][j].tick_params(axis="both", width=0.0)

    for ax in axes.flat:
        for axis in ax.spines:
            ax.spines[axis].set_linewidth(0.75)

    # Add labels
    for i in range(1, irt.dim):
        axes[i][0].set_ylabel(labels[i])
    for j in range(irt.dim):
        axes[-1][j].tick_params(axis="y", width=0.0)
        axes[-1][j].tick_params(axis="x", direction="in", width=0.5)
        axes[-1][j].set_xlabel(labels[j]) 

    for i in range(irt.dim):
        for j in range(i+1, irt.dim):
            axes[i][j].set_axis_off()

    plt.savefig(fname)
    return