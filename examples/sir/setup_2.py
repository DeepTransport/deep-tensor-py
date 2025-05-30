import corner 
from matplotlib import pyplot as plt
import torch 

import deep_tensor as dt

from examples.sir.sir import SIRModel


model = SIRModel()
n_solves = 10_000
params = 2.0 * torch.rand((n_solves, 2))

# TEMP!!
neglogpri = lambda xs: torch.zeros(xs.shape[0])

n_samples = 2_000
xs = torch.randn((n_samples, 2))
ys = model.solve_fwd(model.gauss2unif(xs))
es = model.var_error * torch.randn_like(ys)

yxs = torch.hstack((ys + es, xs))

preconditioner = dt.SampleBasedPreconditioner(yxs)

# rs = preconditioner.reference.random(d=6, n=2_000)
# samples_pre = preconditioner.Q(rs)
# fig = corner.corner(yxs.numpy(), color="tab:red", plot_contours=False)
# corner.corner(samples_pre.numpy(), fig=fig, color="tab:blue", plot_contours=False)
# plt.show()

poly = dt.Lagrange1(num_elems=16)
tt_options = dt.TTOptions(init_rank=12, max_rank=14)

betas = torch.tensor([1e-4]) * torch.tensor(10).sqrt() ** torch.arange(9)
bridge = dt.Tempering(betas)

# TODO: just make a potential function argument
dirt = dt.DIRT(
    negloglik=model.potential, 
    neglogpri=neglogpri, 
    preconditioner=preconditioner,
    bridge=bridge,
    bases=poly,
    tt_options=tt_options
)

xs_test = torch.randn((10, 2))
# xs_test[0] = torch.tensor([0.1, 1.0])
ys_test = model.solve_fwd(model.gauss2unif(xs_test))
ys_test += model.var_error * torch.randn_like(ys_test)

n_rs = 100
rs = dirt.reference.random(d=2, n=n_rs)

n_grid = 300
# xs_grid = torch.linspace(0.04, 0.18, n_grid)
# ys_grid = torch.linspace(0.8, 1.5, n_grid)
xs_grid = torch.linspace(-3.0, 3.0, n_grid)
ys_grid = torch.linspace(-3.0, 3.0, n_grid)
grid = torch.tensor([[x, y] for y in ys_grid for x in xs_grid])

for i, y_i in enumerate(ys_test):

    y_is = y_i.repeat(n_grid**2, 1)
    yx_is = torch.hstack((y_is, grid))

    # Evaluate true conditional density on grid
    neglogfxs_true = model.potential(yx_is)
    fxs_true = torch.exp(-neglogfxs_true).reshape(n_grid, n_grid)

    # Evaluate CIRT density on grid
    neglogfxs_ys = dirt.eval_potential(y_is)
    rs_grid, neglogfxs_grid = dirt.eval_rt(torch.hstack((y_is, grid)))
    neglogfxs_dirt = neglogfxs_grid - neglogfxs_ys
    fxs_dirt = torch.exp(-neglogfxs_dirt).reshape(n_grid, n_grid)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    axes[0].pcolormesh(xs_grid, ys_grid, fxs_true)
    axes[1].pcolormesh(xs_grid, ys_grid, fxs_dirt)
    #axes[1].scatter(*samples.T, c="white", s=2, alpha=0.5)
    #axes[1].scatter(*xs_test[i], c="k")
    plt.show()


    # xs, neglogfxs_dirt = dirt.eval_cirt(y_i, rs)
    # yxs = torch.hstack((y_is, xs))
    # neglogfxs_true = model.potential(yxs)

    # dhell = dt.compute_f_divergence(neglogfxs_dirt, neglogfxs_true, div="h2")
    # print(dhell)

    # samples = model.gauss2unif(xs)

    
    # rs_grid = dirt.eval_rt(grid)[0]
    # _, neglogfxs_grid = dirt.eval_cirt(y_i, rs_grid)

    # neglogpost = model.negloglik_trans(grid, y_i).reshape(n_grid, n_grid)
    # post = torch.exp(-neglogpost)

    # post_dirt = torch.exp(-neglogfxs_grid.reshape(n_grid, n_grid))

    # corner.corner(samples.numpy())
    # plt.show()

# samples[:, -2:] = model.gauss2unif(samples[:, -2:])
# samples = dirt.random(50_000)
# fig = corner.corner(samples.numpy())
# corner.corner(yxs, fig=fig)
# plt.show()
