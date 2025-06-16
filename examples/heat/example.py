#%% Example: Heat equation

import os
from matplotlib import pyplot as plt
import torch

import deep_tensor as dt

from examples.heat import setup_heat_problem
from examples.plotting import add_arrows, plot_dl_function, pairplot


torch.manual_seed(1)
plt.style.use(os.sep.join(["examples", "plotstyle.mplstyle"]))

#%% Generation of models and data 

# Construct the prior, full model and reduced order model
prior, model_full, model_rom = setup_heat_problem()

# Generate true log-diffusion coefficient
xi_true = torch.randn(prior.dim)
logk_true = prior.transform(xi_true)

fig, ax = plt.subplots(figsize=(6.0, 2.0))
cbar_label = r"$\log(\kappa(\bm{x}))$"
plot_dl_function(fig, ax, model_full.vec2func(logk_true), cbar_label)
ax.scatter(*prior.ss.T, s=16, c="k", marker="x")
ax.set_xlabel(r"$x_{0}$")
ax.set_ylabel(r"$x_{1}$")
plt.show()

# Generate true temperature field
u_true = model_full.solve(logk_true)

# Specify magnitude of observation noise
std_error = 1.65e-2
var_error = std_error ** 2

# Extract true temperatures at the observation locations and add 
# observation noise
d_obs = model_full.observe(u_true)
noise = std_error * torch.randn_like(d_obs)
d_obs += noise

fig, ax = plt.subplots(figsize=(6.0, 2.0))
cbar_label = r"$u(\bm{x}, 10)$"
plot_dl_function(fig, ax, model_full.vec2func(u_true[:, -1]), cbar_label, vmin=-0.15, vmax=0.1)
ax.scatter(*model_full.xs_obs.T, s=16, c="k", marker=".")
ax.set_xlabel(r"$x_{0}$")
ax.set_ylabel(r"$x_{1}$")
plt.show()

def neglogpri(xs: torch.Tensor) -> torch.Tensor:
    """Returns the negative log prior density evaluated a given set of 
    samples.
    """
    return 0.5 * xs.square().sum(dim=1)

def _negloglik(model, xs: torch.Tensor) -> torch.Tensor:
    """Returns the negative log-likelihood, for a given model, 
    evaluated at each of a set of samples.
    """
    neglogliks = torch.zeros(xs.shape[0])
    for i, x in enumerate(xs):
        k = prior.transform(x)
        us = model.solve(k)
        d = model.observe(us)
        neglogliks[i] = 0.5 * (d - d_obs).square().sum() / var_error
    return neglogliks

def negloglik_full(xs: torch.Tensor) -> torch.Tensor:
    """Returns the negative log-likelihood for the full model (to be 
    used later).
    """
    return _negloglik(model_full, xs)

def negloglik_rom(xs: torch.Tensor) -> torch.Tensor:
    """Returns the negative log-likelihood for the reduced-order model."""
    return _negloglik(model_rom, xs)

#%% DIRT construction

# Define reference density and preconditioner
reference = dt.GaussianReference()
preconditioner = dt.IdentityMapping(prior.dim, reference)

# Define polynomial basis
poly = dt.Legendre(order=20)

# Reduce the initial and maximum tensor ranks to reduce the cost of each layer
tt_options = dt.TTOptions(init_rank=12, max_rank=12)

# Construct DIRT (this will take a while...)
dirt = dt.DIRT(
    negloglik_rom, 
    neglogpri,
    preconditioner,
    poly, 
    tt_options=tt_options
)

#%% Debiasing

# Generate a set of samples from the DIRT density
rs = dirt.reference.random(d=dirt.dim, n=5000)
xs, potentials_dirt = dirt.eval_irt(rs)

# Evaluate the true potential function (for the full model) at each sample
potentials_exact = neglogpri(xs) + negloglik_full(xs)

# Run independence sampler
res = dt.run_independence_sampler(xs, potentials_dirt, potentials_exact)

print(f"Acceptance rate: {res.acceptance_rate:.2f}")
print(f"Mean IACT (all parameters): {res.iacts.mean():.2f}")
print(f"Maximum IACT (all parameters): {res.iacts.max():.2f}")

# Generate trace plot
parameters = torch.hstack((res.potentials[:, None], res.xs[:, 21:23]))
ylabels = [r"$-\log(f(x))$", r"$\xi_{22}$", r"$\xi_{23}$"]

fig, axes = plt.subplots(1, 3, figsize=(7.5, 3))

for i, ax in enumerate(axes):
    ax.plot(parameters[:, i], c="tab:green", lw=0.5)
    ax.set_ylabel(ylabels[i])
    ax.set_box_aspect(1)
    add_arrows(ax)

axes[1].set_xlabel("Iteration")
plt.show()

# Generate pair plot
xs_pri = dirt.reference.random(d=dirt.dim, n=1000)
xs_post = res.xs[::10]

labels = [r"$\xi_{"+f"{i}"+r"}$" for i in range(22, 25)]
bounds = torch.tensor([[-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0]])

pairplot(
    xs_post[:, 21:24], 
    xs_pri[:, 21:24], 
    truth=xi_true[21:24],
    labels=labels, 
    x_label="Posterior",
    y_label="Prior", 
    bounds=bounds
)