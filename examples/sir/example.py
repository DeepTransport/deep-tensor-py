#%% Example: SIR model

import math
from matplotlib import pyplot as plt
import torch

import deep_tensor as dt

from examples.sir import SIRModel
from examples.plotting import add_arrows


torch.set_default_dtype(torch.float64)
torch.manual_seed(1)

#%% Generation of model and data 

model = SIRModel()

xs_true = torch.tensor([[0.1, 1.0]])
ys_true = model.solve_fwd(xs_true)
noise = torch.randn_like(ys_true)
ys_obs = ys_true + noise

#%% DIRT construction

def negloglik(xs: torch.Tensor) -> torch.Tensor:
    ys = model.solve_fwd(xs)
    return 0.5 * (ys - ys_obs).square().sum(dim=1)

def neglogpri(xs: torch.Tensor) -> torch.Tensor:
    neglogpris = torch.full((xs.shape[0],), -math.log(0.25))
    neglogpris[xs[:, 0] < 0.0] = torch.inf 
    neglogpris[xs[:, 1] > 2.0] = torch.inf
    return neglogpris

def neglogpost(xs: torch.Tensor) -> torch.Tensor:
    return negloglik(xs) + neglogpri(xs)

# Define reference density and preconditioner
bounds = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
reference = dt.GaussianReference()
preconditioner = dt.UniformMapping(bounds, reference)

# Define approximation bases
bases = dt.Legendre(order=30)

# Construct DIRT
dirt = dt.DIRT(neglogpost, preconditioner, bases)

#%% Sampling, Marginalisation and Conditioning

# Define grid to evaluate potential function on
n_grid = 200
beta_grid = torch.linspace(0.05, 0.14, n_grid)
gamma_grid = torch.linspace(0.80, 1.40, n_grid)
grid = torch.tensor([[b, g] for g in gamma_grid for b in beta_grid])

# Evaluate potential function
potentials_grid = dirt.eval_potential(grid)

# Plot DIRT density and true density
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True, sharey=True)

# Compute true density
pdf_true = torch.exp(-(negloglik(grid) + neglogpri(grid)))
pdf_true = pdf_true.reshape(n_grid, n_grid)

# Normalise true density
db = beta_grid[1] - beta_grid[0]
dg = gamma_grid[1] - gamma_grid[0]
pdf_true /= (pdf_true.sum() * db * dg)

# Compute DIRT approximation
pdf_dirt = torch.exp(-potentials_grid)
pdf_dirt = pdf_dirt.reshape(n_grid, n_grid)

axes[0].pcolormesh(beta_grid, gamma_grid, pdf_true)
axes[1].pcolormesh(beta_grid, gamma_grid, pdf_dirt)
axes[0].set_ylabel(r"$\gamma$")
for ax in axes:
    ax.set_xlabel(r"$\beta$")
    ax.set_box_aspect(1)

plt.show()

rs = dirt.reference.random(d=dirt.dim, n=20)
samples, potentials = dirt.eval_irt(rs)

fig, ax = plt.subplots(figsize=(7, 3.5), sharex=True, sharey=True)

ax.pcolormesh(beta_grid, gamma_grid, pdf_dirt)
ax.scatter(*samples.T, c="white", s=4)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\gamma$")
ax.set_box_aspect(1)

plt.show()

# Generate marginal samples of parameter beta
rs_beta = dirt.reference.random(d=1, n=1000)
samples_beta, potentials_beta = dirt.eval_irt(rs_beta, subset="first")

# Evaluate marginal potential on the grid of beta values defined previously
potentials_grid = dirt.eval_potential(beta_grid[:, None], subset="first")

pdf_true_marg = pdf_true.sum(dim=0) * dg
pdf_dirt_marg = torch.exp(-potentials_grid)

fig, ax = plt.subplots(figsize=(6.5, 3.5))

ax.plot(beta_grid, pdf_true_marg, c="k", label=r"True density", zorder=2)
ax.plot(beta_grid, pdf_dirt_marg, c="tab:green", ls="--", label=r"DIRT density", zorder=3)
ax.hist(samples_beta, color="tab:green", density=True, alpha=0.5, zorder=1, label="Samples")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$f(\beta)$")
ax.set_box_aspect(1)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
add_arrows(ax)

plt.show()

# Define beta value to condition on
beta_cond = torch.tensor([[0.10]])

# Generate conditional samples of gamma
rs_cond = dirt.reference.random(d=1, n=1000)
samples_gamma, potentials_gamma = dirt.eval_cirt(beta_cond, rs_cond, subset="first")

# Evaluate conditional potential on a grid of gamma values
gamma_grid = torch.linspace(0.9, 1.1, 200)[:, None]
potentials_grid = dirt.eval_potential_cond(beta_cond, gamma_grid, subset="first")

beta_cond = beta_cond.repeat(gamma_grid.shape[0], 1)
grid_cond = torch.hstack((beta_cond, gamma_grid))
dg = gamma_grid[1] - gamma_grid[0]

# Evaluate true conditional density
pdf_true_cond = torch.exp(-(negloglik(grid_cond) + neglogpri(grid_cond))).flatten()
pdf_dirt_cond = torch.exp(-potentials_grid)

# Normalise true conditional density
pdf_true_cond /= (pdf_true_cond.sum() * dg)

fig, ax = plt.subplots(figsize=(6.5, 3.5))

ax.plot(gamma_grid, pdf_true_cond, c="k", label=r"True density", zorder=3)
ax.plot(gamma_grid, pdf_dirt_cond, c="tab:purple", ls="--", label=r"DIRT density", zorder=3)
ax.hist(samples_gamma, color="tab:purple", density=True, alpha=0.5, zorder=1, label="Samples")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$f(\gamma|\beta=0.1)$")
ax.set_box_aspect(1)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
add_arrows(ax)

plt.show()