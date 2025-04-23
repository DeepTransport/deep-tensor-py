"""Verifies that the Jacobian, dr/dx, of the deep Rosenblatt transport 
is being constructed correctly using a finite difference check.
"""

import time
from torch.linalg import norm

import deep_tensor as dt

from setup import * 


def compute_finite_difference_jac(
    dirt: dt.TTDIRT,
    xs: Tensor,
    subset: str | None = None,
    dx: float = 1e-6
) -> torch.Tensor:
    """Computes a finite difference approximation to the Jacobian."""

    n_xs, d_xs = xs.shape
    dxs = torch.tile(dx * torch.eye(d_xs), (n_xs, 1))

    xs_tiled = torch.tile(xs, (1, d_xs)).reshape(-1, d_xs)
    xs_0 = xs_tiled - dxs 
    xs_1 = xs_tiled + dxs

    zs_0 = dirt.eval_rt(xs_0, subset=subset)[0].T
    zs_1 = dirt.eval_rt(xs_1, subset=subset)[0].T

    J = (zs_1 - zs_0) / (2 * dx)
    return J

headers = [
    "Jacobian Error",
    "Time (s)"
]
headers = [f"{h:16}" for h in headers]

print("")
print(" | ".join(headers))
print("-+-".join(["-" * 16] * len(headers)))

n_zs = 1_000
rs = reference.random(dim, n_zs)

xs = dirt.eval_irt(rs)[0]

t0 = time.time()
Js = dirt.eval_rt_jac(xs)
Js = Js.reshape(dim, dim * n_zs)
t1 = time.time()

Js_fd = compute_finite_difference_jac(dirt, xs)

plt.figure(figsize=(6, 6))
plt.scatter(Js.flatten(), Js_fd.flatten(), s=4)
plt.title("Jacobian Finite Difference Check")
plt.ylabel(r"$J_{\mathrm{fd}}$")
plt.xlabel(r"$J$")
plt.show()

# fname = f"examples/ou_process/figures/04_jacobian_{poly}_{method}_{subset}.png"
# plt.savefig(fname, dpi=500)
# plt.close()

approx_error = norm(Js - Js_fd)

info = [
    f"{approx_error:=16.5e}",
    f"{t1-t0:=16.5f}"
]
print(" | ".join(info))