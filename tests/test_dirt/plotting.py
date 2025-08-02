from matplotlib import pyplot as plt
import torch
from torch import Tensor


def plot_potentials(potentials_dirt: Tensor, potentials_true: Tensor) -> None:

    min_potential = torch.min(potentials_dirt, potentials_true).min()
    max_potential = torch.max(potentials_dirt, potentials_true).max()
    bounds = [min_potential, max_potential]

    plt.figure(figsize=(6, 6))
    plt.scatter(potentials_dirt, potentials_true, s=4, label="Samples")
    plt.plot(bounds, bounds, c="grey", ls="--", label="Expected relationship")
    plt.xlabel("DIRT potential function")
    plt.ylabel("True potential function")
    plt.legend()
    plt.show()
    
    return