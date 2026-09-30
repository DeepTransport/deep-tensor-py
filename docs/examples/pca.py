from typing import Tuple

import torch 
from torch import Tensor


def compute_pca(
    xs: Tensor, 
    eps: float = 1e-2
) -> Tuple[Tensor, Tensor]:
    """Computes a reduced basis ."""
    
    mean = xs.mean(dim=0)
    cov = xs.T.cov()
    vals, vecs = torch.linalg.eigh(cov)

    vals = vals.flip(dims=(0,))
    energies = torch.cumsum(vals, dim=0)
    energies /= energies.max()

    num_components = energies[energies < (1.0-eps)].numel() #+ 1
    basis = vecs[:, -num_components:]

    return mean, basis


def get_pca_weights(ys: Tensor, mean: Tensor, basis: Tensor) -> Tensor:
    """Returns the PCA weights corresponding to sets of observations."""
    return (ys - mean) @ basis


def get_pca_obs(us: Tensor, mean: Tensor, basis: Tensor) -> Tensor:
    """Returns the observations corresponding to sets of PCA weights."""
    return mean + (us @ basis.T)