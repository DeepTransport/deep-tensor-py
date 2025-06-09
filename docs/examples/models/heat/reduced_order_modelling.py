from typing import Tuple

import dolfin as dl
import torch
from torch import Tensor
import tqdm
import ufl


def compute_pod_basis(X: Tensor, tol: float = 0.999) -> Tuple[Tensor, Tensor]:
    """Computes a reduced basis using the POD."""
    
    m = X.mean(dim=1, keepdim=True)
    X -= m

    U, S, V = torch.linalg.svd(X @ X.T)

    energies = S.cumsum(dim=0)
    energies /= energies.max()

    n_components = torch.nonzero(energies > tol)[1]
    print(f"Number of components retained: {int(n_components)}")
    V = U[:, :n_components]
    return m, V


def compute_rom_matrices(
    mesh: dl.RectangleMesh,
    V: Tensor, 
    Vh: dl.FunctionSpace
) -> Tensor:
    """Computes a set of matrices that comprise the reduced stiffness 
    matrix.
    """

    u = dl.TrialFunction(Vh)
    v = dl.TestFunction(Vh)
    z = dl.interpolate(dl.Constant(0.0), Vh)

    d_r = V.shape[1]
    K_rs = torch.zeros((d_r, d_r, mesh.num_vertices()))

    print("Computing stiffness constituent matrices...")

    for i in tqdm.tqdm(range(mesh.num_vertices())):
        
        z_c = z.copy(deepcopy=True)
        z_c.vector()[i] = 1.0
        K = dl.assemble(z_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
        K = torch.tensor(K.array())
        K_rs[:, :, i] = V.T @ K @ V

    return K_rs