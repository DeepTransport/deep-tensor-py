import torch 
from torch import Tensor 
from matplotlib import pyplot as plt


plt.style.use("examples/plotstyle.mplstyle")


def plot_potentials(
    potentials_true: Tensor,
    potentials_dirt: Tensor
) -> None:
    
    min_potential = torch.min(potentials_dirt.min(), potentials_true.min())
    max_potential = torch.max(potentials_dirt.max(), potentials_true.max())
    
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([min_potential, max_potential], [min_potential, max_potential], ls="--", c="k", zorder=1)
    ax.scatter(potentials_true, potentials_dirt, s=5, zorder=2)
    
    ax.set_xlabel(r"$-\log(f(x))$ (True)")
    ax.set_ylabel(r"$-\log(\hat{f}(x))$ (DIRT)")
    ax.set_title("Potential Comparison")

    plt.show()
    return