"""Illustrates the process of building a DIRT."""

from matplotlib import pyplot as plt
import torch
from torch import Tensor

import deep_tensor as dt

from examples.double_banana.double_banana import DoubleBanana


plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)


polys = {
    "lagrange1": dt.Lagrange1(num_elems=50),
    "lagrangep": dt.LagrangeP(order=5, num_elems=10),
    "legendre": dt.Legendre(order=50),
    "fourier": dt.Fourier(order=30)
}

sigma = 0.3
data = torch.tensor([3.0, 5.0])
model = DoubleBanana(sigma, data)

dim = 2
bounds = torch.tensor([-4.0, 4.0])
domain = dt.BoundedDomain(bounds)
reference = dt.GaussianReference(domain)

dx = 1.0
Q_mat = torch.tensor([[dx, 0.0], [0.0, dx]])
Q_mat_inv = torch.tensor([[1.0/dx, 0.0], [0.0, 1.0/dx]])
b = torch.tensor([0.0, 1.0])

def Q(xs: Tensor) -> Tensor:
    ms = xs @ model.Q.T + model.b
    return ms

def Q_inv(ms: Tensor) -> Tensor:
    xs = (ms - model.b) @ model.Q_inv
    return xs

def neglogabsdet_Q_inv(xs: Tensor) -> Tensor:
    n_xs = xs.shape[0]
    neglogabsdet = torch.full((n_xs,), torch.tensor(2*model.dx).log())
    return neglogabsdet

prior = dt.PriorTransformation(
    reference=reference,
    Q=Q, 
    Q_inv=Q_inv, 
    neglogabsdet_Q_inv=neglogabsdet_Q_inv,
    dim=2
)

# references = {
#     "gaussian": dt.GaussianReference(domain=domain),
#     "uniform": dt.UniformReference(domain=domain)
# }

# Build grid for plotting
n = 100
xs_grid = torch.linspace(-4.0, 4.0, n)
ys_grid = torch.linspace(-3.0, 5.0, n)
grid = torch.tensor([[x, y] for x in xs_grid for y in ys_grid])
dx = 8.0 / n

betas = torch.tensor([1e-4, 1e-3, 1e-2, 1e-1, 1.0])

for poly in polys:

    bases = dt.ApproxBases(polys[poly], domain, dim)
    bridge = dt.Tempering(betas=betas)
    dirt = dt.TTDIRT(
        model.negloglik, 
        prior, 
        polys[poly], 
        bridge=bridge
    )

    for k in range(dirt.n_layers):

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

        for ax in axes.flat:
            ax.set_box_aspect(1)

        # Evaluate current approximation to target density
        fxs = dirt.eval_potential(grid, k+1)
        fxs = torch.exp(-fxs)
        axes[0][0].contourf(xs_grid, ys_grid, fxs.reshape(n, n).T, levels=5)
        axes[0][0].set_xlabel("$x_{1}$")
        axes[0][0].set_ylabel("$x_{2}$")
        axes[0][0].set_title(r"$\hat{f}(x)$")

        # Evaluate current target density
        neglogliks = model.negloglik(grid)
        neglogpris = model.neglogpri(grid)
        # neglogliks, neglogpris = model.potential_dirt(grid)
        fxs_true = torch.exp(-neglogliks*dirt.bridge.betas[k]-neglogpris)
        # fxs_true = torch.exp(-neglogliks-neglogpris)
        fxs_true /= (fxs_true.sum() * dx**2)
        axes[0][1].contourf(xs_grid, ys_grid, fxs_true.reshape(n, n).T, levels=5)
        axes[0][1].set_xlabel("$x_{1}$")
        axes[0][1].set_ylabel("$x_{2}$")
        axes[0][1].set_title(r"$f(x)$")

        # Evaluate current approximation to ratio function
        ratio = dirt.irts[k].eval_pdf(grid)
        axes[1][0].contourf(xs_grid, ys_grid, ratio.reshape(n, n).T, levels=5)
        axes[1][0].set_xlabel("$u_{1}$")
        axes[1][0].set_ylabel("$u_{2}$")
        axes[1][0].set_title("Ratio Func. (FTT)")

        # if k > 0:
            
        #     # Evaluate current ratio function
        #     xs = dirt.eval_irt(grid, k)[0]
        #     neglogliks = model.negloglik(xs)
        #     neglogpris = model.neglogpri(xs)
        #     neglogrefs = dirt.reference.eval_potential(grid)[0]
        #     ratio_true = torch.exp(-neglogliks*(dirt.bridge.betas[k]-dirt.bridge.betas[k-1])-neglogrefs)
        #     axes[1][1].contourf(xs_grid, ys_grid, ratio_true.reshape(n, n).T, levels=5)
        #     axes[1][1].set_xlabel("$u_{1}$")
        #     axes[1][1].set_ylabel("$u_{2}$")
        #     axes[1][1].set_title("Ratio Func. (True)")
        
        # else:
        #     axes[1][1].set_axis_off()

        if k == dirt.n_layers - 1:
            samples = dirt.random(100)
            axes[0][0].scatter(*samples.T, c="white", s=0.5)

        figures_dir = "examples/double_banana/figures"
        plt.savefig(f"{figures_dir}/dirt_{poly}_iter_{k}.pdf")
        plt.close()