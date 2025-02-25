"""Illustrates the process of building a DIRT."""

from matplotlib import pyplot as plt
import torch

import deep_tensor as dt

from examples.double_banana.double_banana import DoubleBanana


plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)


polys = {
    "lagrange1": dt.Lagrange1(num_elems=50),
    # "lagrangep": dt.LagrangeP(order=5, num_elems=8),
    # "legendre": dt.Legendre(order=80),
    # "fourier": dt.Fourier(order=30)
}

references = {
    "gaussian": dt.GaussianReference(),
    "uniform": dt.UniformReference()
}

sigma = 0.3
data = torch.tensor([3.0, 5.0])
model = DoubleBanana(sigma, data)

dim = 2
bounds = torch.tensor([-4.0, 4.0])
domain = dt.BoundedDomain(bounds)

# Build grid for plotting
n = 100
xs_grid = torch.linspace(-4.0, 4.0, n)
ys_grid = torch.linspace(-4.0, 4.0, n)
grid = torch.tensor([[x, y] for x in xs_grid for y in ys_grid])
dx = 8.0 / n

for poly in polys:
    for ref in references:

        bases = dt.ApproxBases(polys[poly], domain, dim)
        bridge = dt.Tempering1()
        dirt = dt.TTDIRT(
            model.potential_dirt, 
            bases, 
            bridge=bridge,
            reference=references[ref]
        )

        for k in range(dirt.n_layers):

            fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

            for ax in axes.flat:
                ax.set_box_aspect(1)

            # Evaluate current approximation to target density
            fxs = dirt.eval_potential(grid, k+1)
            fxs = torch.exp(-fxs)
            axes[0][0].contourf(xs_grid, ys_grid, fxs.reshape(n, n).T)
            axes[0][0].set_xlabel("$x_{1}$")
            axes[0][0].set_ylabel("$x_{2}$")
            axes[0][0].set_title(r"$\hat{f}(x)$")

            # Evaluate current target density
            neglogliks, neglogpris = model.potential_dirt(grid)
            fxs_true = torch.exp(-neglogliks*dirt.bridge.betas[k]-neglogpris)
            fxs_true /= (fxs_true.sum() * dx**2)
            axes[0][1].contourf(xs_grid, ys_grid, fxs_true.reshape(n, n).T)
            axes[0][1].set_xlabel("$x_{1}$")
            axes[0][1].set_ylabel("$x_{2}$")
            axes[0][1].set_title(r"$f(x)$")

            # Evaluate current approximation to ratio function
            ratio = dirt.irts[k].eval_pdf(grid)
            axes[1][0].contourf(xs_grid, ys_grid, ratio.reshape(n, n).T)
            axes[1][0].set_xlabel("$u_{1}$")
            axes[1][0].set_ylabel("$u_{2}$")
            axes[1][0].set_title("Ratio Func. (FTT)")

            if k > 0:
                
                # Evaluate current ratio function
                xs = dirt.eval_irt(grid, k)[0]
                neglogliks, neglogpris = model.potential_dirt(xs)
                neglogrefs = -dirt.reference.log_joint_pdf(grid)[0]
                ratio_true = torch.exp(-neglogliks*(dirt.bridge.betas[k]-dirt.bridge.betas[k-1])-neglogrefs)
                axes[1][1].contourf(xs_grid, ys_grid, ratio_true.reshape(n, n).T)
                axes[1][1].set_xlabel("$u_{1}$")
                axes[1][1].set_ylabel("$u_{2}$")
                axes[1][1].set_title("Ratio Func. (True)")
            
            else:
                axes[1][1].set_axis_off()

            figures_dir = "examples/double_banana/figures"
            plt.savefig(f"{figures_dir}/dirt_{poly}_{ref}_iter_{k}.pdf")
            plt.clf()