import torch
from torch import Tensor


def compute_pod_basis(X: Tensor, eps: float = 1.0e-3) -> Tensor:
    """Computes a reduced basis using the POD."""

    U, S, V = torch.linalg.svd(X @ X.T)

    energies = S.cumsum(dim=0)
    energies /= energies.max()

    n_components = torch.sum(energies < 1.0 - eps) + 1
    V = U[:, :n_components]
    return V