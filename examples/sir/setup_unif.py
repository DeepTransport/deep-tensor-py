import corner 
from matplotlib import pyplot as plt
import torch 
from torch import Tensor

import deep_tensor as dt

from examples.sir.sir_unif import SIRModel


plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)


model = SIRModel()

def Q(xs: Tensor) -> Tensor:
    d_xs = xs.shape[1]
    ms = xs * torch.tensor([100.0, 100.0, 100.0, 100.0, 2.0, 2.0])[:d_xs]
    return ms

def Q_inv(ms: Tensor) -> Tensor:
    d_ms = ms.shape[1]
    xs = ms / torch.tensor([100.0, 100.0, 100.0, 100.0, 2.0, 2.0])[:d_ms]
    return xs 

def neglogdet_Q(xs: Tensor) -> Tensor:
    return torch.full((xs.shape[0], ), -torch.tensor(4.0e8).log())

def neglogdet_Q_inv(ms: Tensor) -> Tensor:
    return torch.full((ms.shape[0], ), -torch.tensor(0.25e-8).log())

reference = dt.UniformReference()

preconditioner = dt.IdentityPreconditioner(
    reference=reference, 
    Q=Q, 
    Q_inv=Q_inv,
    neglogdet_Q=neglogdet_Q,
    neglogdet_Q_inv=neglogdet_Q_inv,
    dim=6
)

poly = dt.Lagrange1(num_elems=30)
tt_options = dt.TTOptions(init_rank=20, max_rank=30)

betas = torch.tensor([1e-4]) * torch.tensor(10).sqrt() ** torch.arange(9)
bridge = dt.Tempering(betas)
print(betas)

dirt = dt.DIRT(
    negloglik=model.negloglik, 
    neglogpri=model.neglogpri, 
    preconditioner=preconditioner,
    bridge=bridge,
    bases=poly,
    tt_options=tt_options
)

# samples = dirt.random(50_000)
# corner.corner(samples.numpy())
# plt.show()

xs_test = 2.0 * torch.rand((10, 2))
xs_test[0] = torch.tensor([0.1, 1.0])
ys_test = model.solve_fwd(xs_test)
# ys_test += model.var_error * torch.randn_like(ys_test)

print(ys_test)

# For DHell
n_rs = 50_000
rs = dirt.reference.random(d=2, n=n_rs)

n_grid = 100
# xs_grid = torch.linspace(0.04, 0.18, n_grid)
# ys_grid = torch.linspace(0.8, 1.5, n_grid)
xs_grid = torch.linspace(0.0, 2.0, n_grid)
ys_grid = torch.linspace(0.0, 2.0, n_grid)
grid = torch.tensor([[x, y] for y in ys_grid for x in xs_grid])

for i, y_i in enumerate(ys_test):

    print(y_i)

    y_is = y_i.repeat(n_grid**2, 1)
    yx_is = torch.hstack((y_is, grid))

    # Evaluate true conditional density on grid
    neglogfxs_true = model.negloglik(yx_is) + model.neglogpri(yx_is)
    fxs_true = torch.exp(-neglogfxs_true).reshape(n_grid, n_grid)

    # Evaluate CIRT density on grid
    neglogfxs_ys = dirt.eval_potential(y_is)
    rs_grid, neglogfxs_grid = dirt.eval_rt(torch.hstack((y_is, grid)))
    neglogfxs_dirt = neglogfxs_grid - neglogfxs_ys
    fxs_dirt = torch.exp(-neglogfxs_dirt).reshape(n_grid, n_grid)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

    axes[0].pcolormesh(xs_grid, ys_grid, fxs_true, rasterized=True)
    axes[1].pcolormesh(xs_grid, ys_grid, fxs_dirt, rasterized=True)
    axes[0].set_title(r"$f(x)$ (True)")
    axes[1].set_title(r"$\hat{f}(x)$ (DIRT)")
    axes[0].set_ylabel(r"$x_{1}$")

    for ax in axes:
        ax.scatter(*xs_test[i], c="k", marker="x", s=5)
        ax.set_xlabel(r"$x_{0}$")

    xs, neglogfxs_dirt = dirt.eval_cirt(y_i, rs)
    yxs = torch.hstack((y_i.repeat(n_rs, 1), xs))
    neglogfxs_true = model.negloglik(yxs) + model.neglogpri(yxs)

    dhell2 = dt.compute_f_divergence(-neglogfxs_dirt, -neglogfxs_true, div="h2")
    dhell = dhell2.sqrt()
    print(f"{i}: {dhell}")

    # axes[1].scatter(*xs[:100].T, c="white", s=1)

    plt.suptitle(r"$\mathcal{D}_{\mathrm{H}}$"+f": {dhell:.4f}")
    plt.savefig(f"examples/sir/figures/posterior_{i}_unif.pdf")

