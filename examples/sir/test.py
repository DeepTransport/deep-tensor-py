import corner 
from matplotlib import pyplot as plt
import torch 

import deep_tensor as dt

from examples.sir.sir import SIRModel
import examples.plotting as plotting

plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)

model = SIRModel()
n_solves = 10_000
params = 2.0 * torch.rand((n_solves, 2))

n_samples = 2_000
xs = torch.randn((n_samples, 2))
ys = model.solve_fwd(model.gauss2unif(xs))
es = model.var_error * torch.randn_like(ys)

samples_joint = torch.hstack((ys + es, xs))
preconditioner = dt.SampleBasedPreconditioner(samples_joint)

samples_reference = preconditioner.reference.random(d=6, n=2_000)
samples_preconditioner = preconditioner.Q(samples_reference)
samples_joint_pullback = preconditioner.Q_inv(samples_joint)

labels = [f"$y_{i+1}$" for i in range(4)] + [f"$x_{i+1}$" for i in range(2)]

plotting.corner_plot(
    xs=samples_preconditioner, 
    ys=samples_joint, 
    x_label=r"Samples from $Q_{\sharp}\rho(x)$",
    y_label=r"Samples from $f(x)$",
    labels=labels,
    fname="examples/sir/figures/preconditioner.pdf"
)

plotting.corner_plot(
    xs=samples_reference, 
    ys=samples_joint_pullback, 
    x_label=r"Samples from $\rho(x)$",
    y_label=r"Samples from $Q^{\sharp}f(x)$",
    labels=labels,
    fname="examples/sir/figures/preconditioner_pullback.pdf"
)

poly = dt.Lagrange1(num_elems=17)
tt_options = dt.TTOptions(init_rank=17, max_rank=17, tt_method="amen")

betas = torch.tensor([1e-4]) * torch.tensor(10).sqrt() ** torch.arange(9)
# betas = betas[::2]
# betas = torch.tensor([1e-4]) * torch.tensor(10) ** (0.25 * torch.arange(17))
bridge = dt.Tempering(betas)
print(betas)
# bridge = dt.Tempering(ess_tol=0.7)

# tt_options = dt.TTOptions(max_cross=2)

dirt = dt.DIRT(
    negloglik=model.negloglik_joint, 
    neglogpri=model.neglogpri_joint, 
    preconditioner=preconditioner,
    bridge=bridge,
    bases=poly,
    tt_options=tt_options
)

samples_dirt = dirt.random(2000)
samples_dirt_pullback = preconditioner.Q_inv(samples_dirt)

plotting.corner_plot(
    xs=samples_dirt, 
    ys=samples_joint, 
    x_label=r"Samples from $\hat{f}(x)$",
    y_label=r"Samples from $f(x)$",
    labels=labels,
    fname="examples/sir/figures/dirt.pdf"
)

plotting.corner_plot(
    xs=samples_dirt_pullback, 
    ys=samples_joint_pullback, 
    x_label=r"Samples from $Q^{\sharp}\hat{f}(x)$",
    y_label=r"Samples from $Q^{\sharp}f(x)$",
    labels=labels,
    fname="examples/sir/figures/dirt_pullback.pdf"
)

xs_test = torch.randn((10, 2))
xs_test[0] = model.unif2gauss(torch.tensor([0.1, 1.0]))
ys_test = model.solve_fwd(model.gauss2unif(xs_test))
# ys_test += model.var_error * torch.randn_like(ys_test)  # add error

# For DHell
n_rs = 50_000
rs = dirt.reference.random(d=2, n=n_rs)

def plot_density_comparison(xs_grid, ys_grid, fxs_true, fxs_dirt, x_true, dhell, fname):

    fig, axes = plt.subplots(
        nrows=1, ncols=2, 
        figsize=(8, 4), 
        sharex=True, sharey=True
    )
    
    axes[0].pcolormesh(xs_grid, ys_grid, fxs_true, rasterized=True)
    axes[1].pcolormesh(xs_grid, ys_grid, fxs_dirt, rasterized=True)
    axes[0].set_title(r"$f(x)$ (True)")
    axes[1].set_title(r"$\hat{f}(x)$ (DIRT)")
    axes[0].set_ylabel(r"$x_{1}$")

    for ax in axes:
        ax.scatter(*x_true, c="k", marker="x", s=5)
        ax.set_xlabel(r"$x_{0}$")

    plt.suptitle(r"$\mathcal{D}_{\mathrm{H}}$"+f": {dhell:.4f}")
    plt.savefig(fname)
    
    return

n_grid = 100

for i, y_i in enumerate(ys_test):

    if i == 0:
        xs_grid = torch.linspace(-1.4, -0.9, n_grid)
        ys_grid = torch.linspace(-0.5, 0.5, n_grid)
    else:
        xs_grid = torch.linspace(-3.0, 3.0, n_grid)
        ys_grid = torch.linspace(-3.0, 3.0, n_grid)

    dx = xs_grid[1] - xs_grid[0]
    grid = torch.tensor([[x, y] for y in ys_grid for x in xs_grid])

    y_is = y_i.repeat(n_grid**2, 1)
    yx_is = torch.hstack((y_is, grid))

    # Evaluate true conditional density on grid
    neglogfxs_true = model.potential_joint(yx_is)
    fxs_true = torch.exp(-neglogfxs_true) / (neglogfxs_true.sum() * dx**2)
    fxs_true = fxs_true.reshape(n_grid, n_grid)

    # Evaluate CIRT density on grid
    neglogfxs_ys = dirt.eval_potential(y_is)
    rs_grid, neglogfxs_grid = dirt.eval_rt(torch.hstack((y_is, grid)))
    neglogfxs_dirt = neglogfxs_grid - neglogfxs_ys
    fxs_dirt = torch.exp(-neglogfxs_dirt).reshape(n_grid, n_grid)

    # Estimate Hellinger distance
    xs, neglogfxs_dirt = dirt.eval_cirt(y_i, rs)
    yxs = torch.hstack((y_i.repeat(n_rs, 1), xs))
    neglogfxs_true = model.potential_joint(yxs)

    dhell2 = dt.compute_f_divergence(-neglogfxs_dirt, -neglogfxs_true, div="h2")
    dhell = dhell2.sqrt()
    print(f"{i}: {dhell}")

    plot_density_comparison(
        xs_grid, 
        ys_grid, 
        fxs_true, 
        fxs_dirt, 
        xs_test[i], 
        dhell=dhell, 
        fname=f"examples/sir/figures/posterior_{i}.pdf"
    )