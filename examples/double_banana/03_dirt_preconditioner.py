"""A demonstration of how to use a previously constructed DIRT object 
as a preconditioner for subsequent inference tasks.
"""

from matplotlib import pyplot as plt
import torch

import deep_tensor as dt

from examples.double_banana.double_banana import DoubleBanana


plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)


poly = dt.Lagrange1(num_elems=30)
# poly = dt.Lagrange1(num_elems=20)

sigma = 0.3
data = torch.tensor([3.0, 5.0])
model = DoubleBanana(sigma, data)

dim = 2
reference = dt.GaussianReference()

preconditioner = dt.IdentityPreconditioner(dim, reference)

betas = torch.tensor([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
bridge = dt.Tempering(betas=betas)

dirt_pre = dt.DIRT(
    model.neglogpri,
    model.negloglik, 
    preconditioner, 
    poly, 
    bridge=bridge,
    # tt_options=dt.TTOptions(verbose=False),
    # dirt_options=dt.DIRTOptions(method="eratio")
) 

# dirt_pre.save("test")
# dirt_pre = dt.SavedDIRT("test", preconditioner)

preconditioner_dirt = dt.DIRTPreconditioner(dirt_pre)

dirt = dt.DIRT(
    model.neglogpri,
    model.negloglik, 
    preconditioner_dirt, 
    poly, 
    bridge=dt.SingleLayer(),
    # dirt_options=dt.DIRTOptions(method="eratio")
)

rs = reference.random(d=2, n=50_000)

xs_pre, neglogfxs_pre = dirt_pre.eval_irt(rs)
xs, neglogfxs = dirt.eval_irt(rs)

neglogfxs_pre_exact = model.negloglik(xs_pre) + model.neglogpri(xs_pre)
neglogfxs_exact = model.negloglik(xs) + model.neglogpri(xs)

# print(xs)
# print(neglogfxs_pre)
# print(neglogfxs)
# print(neglogfxs_exact)
# print(neglogfxs-neglogfxs_exact)

# Ideally, the acceptance rate should be higher once we have built an 
# extra DIRT
res = dt.run_independence_sampler(xs_pre, neglogfxs_pre, neglogfxs_pre_exact)
print(res.acceptance_rate)
res = dt.run_independence_sampler(xs, neglogfxs, neglogfxs_exact)
print(res.acceptance_rate)

# import corner

# corner.corner(xs_pre.numpy())
# plt.savefig("layer0.pdf")
# corner.corner(xs.numpy())
# plt.savefig("layer1.pdf")
 
# Build grid for plotting
n = 100
xs_grid = torch.linspace(-3.9, 3.9, n)
ys_grid = torch.linspace(-3.9, 3.9, n)
grid = torch.tensor([[x, y] for y in ys_grid for x in xs_grid])
dx = 8.0 / n

dirts = [dirt_pre, dirt]

for i, dirt in enumerate(dirts):

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    samples = dirt.random(100)

    # Evaluate current approximation to target density
    fxs = dirt.eval_pdf(grid)
    # fxs = torch.exp(-fxs)
    axes[0].contourf(xs_grid, ys_grid, fxs.reshape(n, n), levels=5)
    axes[0].set_xlabel("$x_{1}$")
    axes[0].set_ylabel("$x_{2}$")
    axes[0].set_title(r"$\hat{f}(x)$")
    axes[0].scatter(*samples.T, c="white", s=2)

    # Evaluate current target density
    neglogliks = model.negloglik(grid)
    neglogpris = model.neglogpri(grid)
    fxs_true = torch.exp(-neglogliks-neglogpris)
    fxs_true /= (fxs_true.sum() * dx**2)
    axes[1].contourf(xs_grid, ys_grid, fxs_true.reshape(n, n), levels=5)
    axes[1].set_xlabel("$x_{1}$")
    axes[1].set_ylabel("$x_{2}$")
    axes[1].set_title(r"$f(x)$")

    figures_dir = "examples/double_banana/figures"
    plt.savefig(f"{figures_dir}/03_dirt_{i}.pdf")
    plt.close()