"""Verifies that the Jacobian, dz/dx, of the Rosenblatt transport is 
being constructed correctly using a finite difference check.
"""

import time
from torch.linalg import norm

import deep_tensor as dt

from examples.ou_process.setup_ou import * 


def compute_finite_difference_jac(
    sirt: dt.TTSIRT,
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

    zs_0 = sirt.eval_rt(xs_0, subset).T
    zs_1 = sirt.eval_rt(xs_1, subset).T

    J = (zs_1 - zs_0) / (2 * dx)
    return J

headers = [
    "Polynomial",
    "Method",
    "Subset",
    "Jacobian Error",
    "Time (s)"
]
headers = [f"{h:16}" for h in headers]

print("")
print(" | ".join(headers))
print("-+-".join(["-" * 16] * len(headers)))

subsets = ["first", "last"]
methods = ["manual", "autodiff"]

n_zs = 1000
zs = torch.rand((n_zs, dim))

for poly in polys_dict:
    for method in methods:
        for subset in subsets:

            sirt: dt.TTSIRT = sirts[poly]["random"]

            xs = sirt.eval_irt(zs)[0]

            t0 = time.time()
            Js = sirt.eval_rt_jac(xs, method, subset)
            Js = Js.reshape(dim, dim * n_zs)
            t1 = time.time()

            # Js_fd = sirt.eval_rt_jac(xs, method="autodiff", subset=subset).reshape(dim, dim*n_zs)
            Js_fd = compute_finite_difference_jac(sirt, xs, subset)

            plt.figure(figsize=(6, 6))
            plt.scatter(Js.flatten(), Js_fd.flatten(), s=4)
            plt.title("Jacobian Finite Difference Check")
            plt.ylabel(r"$J_{\mathrm{fd}}$")
            plt.xlabel(r"$J$")

            fname = f"examples/ou_process/figures/04_jacobian_{poly}_{method}_{subset}.png"
            plt.savefig(fname, dpi=500)
            plt.close()

            approx_error = norm(Js - Js_fd)

            info = [
                f"{poly:16}",
                f"{method:16}",
                f"{subset:16}",
                f"{approx_error:=16.5e}",
                f"{t1-t0:=16.5f}"
            ]
            print(" | ".join(info))