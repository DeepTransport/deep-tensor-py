"""Illustrates the process of building a DIRT."""

from matplotlib import pyplot as plt
import torch

import deep_tensor as dt

from examples.double_banana.double_banana import DoubleBanana


plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)


polys = {
    "chebyshev1st": dt.Chebyshev1st(order=30),
    "chebyshev2nd": dt.Chebyshev2nd(order=30),
    "lagrange1": dt.Lagrange1(num_elems=30),
    "lagrangep": dt.LagrangeP(order=5, num_elems=6),
    "legendre": dt.Legendre(order=30),
    "fourier": dt.Fourier(order=20)
}

sigma = 0.3
data = torch.tensor([3.0, 5.0])
model = DoubleBanana(sigma, data)

dim = 2
reference = dt.GaussianReference()

Q = lambda xs: xs 
Q_inv = lambda ms: ms 
neglogdet_Q = lambda xs: torch.zeros((xs.shape[0],))
neglogdet_Q_inv = lambda ms: torch.zeros((ms.shape[0],))

preconditioner = dt.Preconditioner(
    reference=reference,
    Q=Q, 
    Q_inv=Q_inv, 
    neglogdet_Q=neglogdet_Q,
    neglogdet_Q_inv=neglogdet_Q_inv,
    dim=2
)

# Build grid for plotting
n = 100
xs_grid = torch.linspace(-4.0, 4.0, n)
ys_grid = torch.linspace(-4.0, 4.0, n)
grid = torch.tensor([[x, y] for y in ys_grid for x in xs_grid])
dx = 8.0 / n

for poly in polys:

    betas = torch.tensor([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    bridge = dt.Tempering(betas=betas)

    dirt = dt.DIRT(
        model.neglogpri,
        model.negloglik, 
        preconditioner, 
        polys[poly], 
        bridge=bridge,
        tt_options=dt.TTOptions(verbose=False),
        dirt_options=dt.DIRTOptions(verbose=False)
    )

    for k in range(dirt.n_layers):

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

        for ax in axes.flat:
            ax.set_box_aspect(1)

        # Evaluate current approximation to target density
        fxs = dirt.eval_pdf(grid, k+1)
        # fxs = torch.exp(-fxs)
        axes[0][0].contourf(xs_grid, ys_grid, fxs.reshape(n, n), levels=5)
        axes[0][0].set_xlabel("$x_{1}$")
        axes[0][0].set_ylabel("$x_{2}$")
        axes[0][0].set_title(r"$\hat{f}(x)$")

        # Evaluate current target density
        neglogliks = model.negloglik(grid)
        neglogpris = model.neglogpri(grid)
        fxs_true = torch.exp(-neglogliks*dirt.bridge.betas[k]-neglogpris)
        fxs_true /= (fxs_true.sum() * dx**2)
        axes[0][1].contourf(xs_grid, ys_grid, fxs_true.reshape(n, n), levels=5)
        axes[0][1].set_xlabel("$x_{1}$")
        axes[0][1].set_ylabel("$x_{2}$")
        axes[0][1].set_title(r"$f(x)$")

        # Evaluate current approximation to ratio function
        ratio = dirt.sirts[k].eval_pdf(grid)
        axes[1][0].contourf(xs_grid, ys_grid, ratio.reshape(n, n), levels=5)
        axes[1][0].set_xlabel("$u_{1}$")
        axes[1][0].set_ylabel("$u_{2}$")
        axes[1][0].set_title("Ratio Func. (FTT)")

        if k > 0:
            
            # Evaluate current ratio function
            xs = dirt.eval_irt(grid, k)[0]
            neglogliks = model.negloglik(xs)
            neglogpris = model.neglogpri(xs)
            neglogrefs = dirt.reference.eval_potential(grid)[0]
            ratio_true = torch.exp(-neglogliks*(dirt.bridge.betas[k]-dirt.bridge.betas[k-1])-neglogrefs)
            axes[1][1].contourf(xs_grid, ys_grid, ratio_true.reshape(n, n), levels=5)
            axes[1][1].set_xlabel("$u_{1}$")
            axes[1][1].set_ylabel("$u_{2}$")
            axes[1][1].set_title("Ratio Func. (True)")
        
        else:
            axes[1][1].set_axis_off()

        if k == dirt.n_layers - 1:
            samples = dirt.random(100)
            axes[0][0].scatter(*samples.T, c="white", s=0.5)

        figures_dir = "examples/double_banana/figures"
        plt.savefig(f"{figures_dir}/01_dirt_{poly}_iter_{k}.pdf")
        plt.close()