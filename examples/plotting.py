from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import torch 
from torch import Tensor


def corner_plot(
    xs: Tensor,
    ys: Tensor | None = None,
    labels: list[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    fname: str | None = None
) -> None:
    """Generates a corner plot for a set of joint distributions."""

    dim = xs.shape[1]

    if labels is None:
        # Use default parameter labels
        labels = [f"$x_{i+1}$" for i in range(dim)]

    fig, axes = plt.subplots(
        dim, 
        dim, 
        figsize=(2*dim, 2*dim),
        sharex="col"
    )

    if ys is not None:
        xys = torch.vstack((xs, ys))
    else:
        xys = xs.clone()

    samples_min = xys.min(dim=0).values
    samples_max = xys.max(dim=0).values
    samples_range = samples_max - samples_min

    lims_min = samples_min - 0.05 * samples_range
    lims_max = samples_max + 0.05 * samples_range
    
    bounds = torch.hstack((lims_min[:, None], lims_max[:, None]))

    for ax in axes.flat:
        ax.set_box_aspect(1)

    for i in range(dim):
        for j in range(i+1):

            if i == j:
                
                # Plot marginal
                xs_kde = torch.linspace(*bounds[j], 100)
                density = gaussian_kde(xs[:, j])(xs_kde)
                axes[i][j].plot(xs_kde, density, c="tab:blue", lw=1.5, label=x_label)
                max_density = max(density)

                if ys is not None:
                    density_ys = gaussian_kde(ys[:, j])(xs_kde)
                    axes[i][j].plot(xs_kde, density_ys, c="tab:orange", lw=1.5, label=y_label)
                    max_density = max(max_density, max(density_ys))

                axes[i][j].yaxis.set_ticklabels([])
                axes[i][j].set_ylim(0.0, 1.1 * max_density)
            
            else:

                axes[i][j].set_xlim(*bounds[j])
                axes[i][j].set_ylim(*bounds[i])
                axes[i][j].scatter(xs[:, j], xs[:, i], s=4, c="tab:blue", alpha=0.5)
                # axes[i][j].contour(xs, ys, kde_zs, levels=12, linewidths=1.5)
                if j > 0:
                    axes[i][j].yaxis.set_ticklabels([])
                
                if ys is not None:
                    axes[i][j].scatter(ys[:, j], ys[:, i], s=4, c="tab:orange", alpha=0.5)
            
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

    plt.savefig(fname)
    return