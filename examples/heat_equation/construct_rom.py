"""Constructs a reduced-order model for the heat equation, using the 
POD and DEIM.
"""

from typing import Tuple


from setup import *
from plotting import *


n_snapshots = 1000

m_samples = torch.randn((n_snapshots, prob.prior.dim))


def get_state_matrix(u: hl.TimeDependentVector) -> Tensor:
    return torch.hstack([torch.tensor(u_i[:])[:, None] for u_i in u.data])


def compute_pod_basis(X: Tensor, tol: float = 0.99) -> Tuple[Tensor, Tensor]:
    """Computes a reduced basis using the POD."""
    
    m = X.mean(dim=1, keepdim=True)
    X -= m

    U, S, V = torch.linalg.svd(X @ X.T)

    energies = S.cumsum(dim=0)
    energies /= energies.max()

    n_components = torch.nonzero(energies > tol)[0]
    print(f"Number of components retained: {n_components}")
    V = U[:, :n_components]
    return m, V


snapshots = []

for i in range(n_snapshots):
    x = prob.generate_vector()
    x[hl.PARAMETER] = m_samples[i]
    prob.solveFwd(x[hl.STATE], x)
    snapshots.append(get_state_matrix(x[hl.STATE]))

snapshots = torch.hstack(snapshots)

print("Computing POD basis...")
m, V = compute_pod_basis(snapshots)
    
model_r = HeatSolverROM(prob, m, V)

# TODO: change n_snapshots?
m_samples_validation = torch.randn((n_snapshots, prob.prior.dim))

for i in range(100):

    x = prob.generate_vector()
    x[hl.PARAMETER] = m_samples_validation[i]
    
    t0 = time.time()
    prob.solveFwd(x[hl.STATE], x)
    u_full = get_state_matrix(x[hl.STATE])
    t1 = time.time()
    print(t1-t0)

    t0 = time.time()
    u_rom = model_r.solveFwd(m_samples_validation[i])
    t1 = time.time()
    print(t1-t0)
    
    print("--")
    
    u_full = prob.vec2func(u_full[:, -1], hl.STATE)
    u_rom = prob.vec2func(u_rom[:, -1], hl.STATE)

    # plot_function(u_full, vmin=-0.2, vmax=0.2)
    # plt.show()
    # plot_function(u_rom, vmin=-0.2, vmax=0.2)
    # plt.show()

# Verify that the ROM produces good results by drawing some samples 
# from the prior and comparing the data from the ROM to the 
# full model