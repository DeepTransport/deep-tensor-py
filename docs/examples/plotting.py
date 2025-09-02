from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import torch 
from torch import Tensor 


STYLE_PATH = Path(__file__).resolve().parent.joinpath("plotstyle.mplstyle")


def set_plot_style():
    plt.style.use(STYLE_PATH)


def add_arrows(ax) -> None:
    """Removes the top spine and right spine from a plot, and adds 
    arrows to the ends of the bottom spine and left spine.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(1, 0, ">k", markersize=6, transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", markersize=6, transform=ax.transAxes, clip_on=False)
    return


def pairplot(
    xs: Tensor,
    ys: Tensor | None = None,
    truth: Tensor | None = None,
    labels: List[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    bounds: Tensor | None = None
) -> None:
    """Generates a pair plot for a set of joint distributions."""

    dim = xs.shape[1]

    if labels is None:
        labels = [f"$x_{i+1}$" for i in range(dim)]

    if ys is not None:
        xys = torch.vstack((xs, ys))
    else:
        xys = xs.clone()

    if bounds is None:
        samples_min = xys.min(dim=0).values
        samples_max = xys.max(dim=0).values
        samples_range = samples_max - samples_min
        lims_min = samples_min - 0.05 * samples_range
        lims_max = samples_max + 0.05 * samples_range
        bounds = torch.hstack((lims_min[:, None], lims_max[:, None]))

    figwidth = min(2*dim, 7)
    fig, axes = plt.subplots(dim, dim, figsize=(figwidth, figwidth), sharex="col")

    for ax in axes.flat:
        ax.set_box_aspect(1)

    for i in range(dim):
        for j in range(i+1):

            if i == j:  # Plot marginals
                
                xs_kde = torch.linspace(*bounds[j], steps=100)
                density = gaussian_kde(xs[:, j])(xs_kde)
                axes[i][j].plot(xs_kde, density, c="tab:red", lw=1.5, label=x_label)
                max_density = max(density)
                
                if ys is not None:
                    density_ys = gaussian_kde(ys[:, j])(xs_kde)
                    axes[i][j].plot(xs_kde, density_ys, c="tab:grey", lw=1.5, label=y_label)
                    max_density = max(max_density, max(density_ys))
                
                axes[i][j].yaxis.set_ticklabels([])
                axes[i][j].set_ylim(0.0, 1.1 * max_density)

                if truth is not None:
                    axes[i][j].axvline(truth[j], c="k", lw=1.5, ls="--", label="Truth")
            
            else:

                if truth is not None:
                    axes[i][j].axvline(truth[j], c="k", ls="--", lw=1.5, zorder=2)
                    axes[i][j].axhline(truth[i], c="k", ls="--", lw=1.5, zorder=2)
                    axes[i][j].scatter(truth[j], truth[i], c="k", marker="s", s=20, zorder=3)

                axes[i][j].set_xlim(*bounds[j])
                axes[i][j].set_ylim(*bounds[i])
                axes[i][j].scatter(xs[:, j], xs[:, i], s=4, c="tab:red", alpha=0.5, zorder=1)
                
                if j > 0:
                    axes[i][j].yaxis.set_ticklabels([])
                
                if ys is not None:
                    axes[i][j].scatter(ys[:, j], ys[:, i], s=4, c="tab:grey", alpha=0.5, zorder=0)
            
            if i == dim-1:
                axes[i][j].tick_params(axis="y", width=0.0)
                axes[i][j].tick_params(axis="x", direction="in", width=0.5)
                axes[i][j].set_xlabel(labels[j]) 
            else:
                axes[i][j].tick_params(axis="both", width=0.0)

    for ax in axes.flat:
        for axis in ax.spines:
            ax.spines[axis].set_linewidth(0.75)

    # Add labels
    for i in range(1, dim):
        axes[i][0].set_ylabel(labels[i])
    for j in range(dim):
        axes[-1][j].tick_params(axis="y", width=0.0)
        axes[-1][j].tick_params(axis="x", direction="in", width=0.5)
        axes[-1][j].set_xlabel(labels[j]) 
    
    handles, labels = axes[-1][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    for i in range(dim):
        for j in range(i+1, dim):
            axes[i][j].set_axis_off()

    plt.show()
    return