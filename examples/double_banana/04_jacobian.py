"""Verifies that the Jacobian, dz/dx, of the Rosenblatt transport is 
being constructed correctly using a finite difference check.
"""

import time

from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch.linalg import norm

import deep_tensor as dt

from double_banana import DoubleBanana

# torch.autograd.set_detect_anomaly(True)


plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)


def compute_finite_difference_jac(
    dirt: dt.DIRT,
    xs: Tensor,
    subset: str,
    dx: float = 1e-6
) -> torch.Tensor:
    """Computes a finite difference approximation to the Jacobian."""

    n_xs, d_xs = xs.shape
    dxs = torch.tile(dx * torch.eye(d_xs), (n_xs, 1))

    xs_tiled = torch.tile(xs, (1, d_xs)).reshape(-1, d_xs)
    xs_0 = xs_tiled - dxs 
    xs_1 = xs_tiled + dxs

    zs_0 = dirt.eval_irt(xs_0, subset=subset)[0].T
    zs_1 = dirt.eval_irt(xs_1, subset=subset)[0].T

    J = (zs_1 - zs_0) / (2 * dx)
    return J


def plot_jacobian_comparison(Js: Tensor, Js_fs: Tensor, fname: str) -> None:

    J_min = torch.min(Js.min(), Js_fs.min())
    J_max = torch.max(Js.max(), Js_fd.max())
    
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([J_min, J_max], [J_min, J_max], ls="--", c="k", zorder=1)
    ax.scatter(Js.flatten(), Js_fd.flatten(), s=4)
    
    ax.set_title("Jacobian Finite Difference Check")
    ax.set_ylabel(r"$J_{\mathrm{fd}}$")
    ax.set_xlabel(r"$J$")
    
    plt.savefig(fname, dpi=500)
    plt.close()

    return


polys = {
    # "chebyshev1st": dt.Chebyshev1st(order=30),
    # "chebyshev2nd": dt.Chebyshev2nd(order=30),
    "lagrange1": dt.Lagrange1(num_elems=10),
    # "lagrangep": dt.LagrangeP(order=2, num_elems=3),
    # "legendre": dt.Legendre(order=30),
    # "fourier": dt.Fourier(order=20)
}

sigma = 0.3
data = torch.tensor([3.0, 5.0])
model = DoubleBanana(sigma, data)

dim = 2
reference = dt.GaussianReference()

preconditioner = dt.IdentityPreconditioner(dim, reference)

headers = [
    "Polynomial",
    "Subset",
    "Jacobian Error",
    "Time (s)"
]
headers = [f"{h:16}" for h in headers]

print("")
print(" | ".join(headers))
print("-+-".join(["-" * 16] * len(headers)))

subsets = ["first", "last"]

n_zs = 1000
zs = preconditioner.reference.random(d=dim, n=n_zs)

for poly in polys:
    for subset in subsets:

        dirt = dt.DIRT(
            model.negloglik,
            model.neglogpri, 
            preconditioner,
            polys[poly],
            tt_options=dt.TTOptions(verbose=False),
            dirt_options=dt.DIRTOptions(verbose=False)
        )

        xs = dirt.eval_rt(zs)[0]

        t0 = time.time()
        Js = dirt.eval_irt_jac(zs, subset)
        Js = Js.reshape(dim, dim * n_zs)
        print(Js.isnan().sum())
        t1 = time.time()

        # Js_fd = sirt.eval_rt_jac(xs, method="autodiff", subset=subset).reshape(dim, dim*n_zs)
        Js_fd = compute_finite_difference_jac(dirt, zs, subset)

        print(Js_fd[Js.isnan()])

        fname = f"examples/double_banana/figures/04_jacobian_{poly}_{subset}.png"
        plot_jacobian_comparison(Js, Js_fd, fname)

        approx_error = norm(Js - Js_fd)

        info = [
            f"{poly:16}",
            f"{subset:16}",
            f"{approx_error:=16.5e}",
            f"{t1-t0:=16.5f}"
        ]
        print(" | ".join(info))