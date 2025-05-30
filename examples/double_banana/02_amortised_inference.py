"""Builds a DIRT approximation to the joint density of the observations 
and parameters, then draws samples from (conditional) posteriors 
associated with varying data.
"""

import corner
from matplotlib import pyplot as plt
import torch

import deep_tensor as dt

from examples.double_banana.double_banana import DoubleBanana


plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)


poly = dt.Lagrange1(num_elems=40)

sigma = 0.3
data = torch.tensor([3.0, 5.0])
model = DoubleBanana(sigma, data)

# Build linear sample-based preconditioner
n_xs = 1000
xs = torch.normal(0.0, 1.0, size=(n_xs, 2))
ys = model.param_to_obs(xs)

preconditioner = dt.SampleBasedPreconditioner(xs, ys)

bridge = dt.Tempering()

dirt = dt.DIRT(
    model.neglogpri_joint,
    model.negloglik_joint, 
    preconditioner, 
    poly, 
    bridge=bridge
)

# Plot the joint distribution
samples = dirt.random(50_000)
corner.corner(samples.numpy(), labels=[r"$y$", r"$x_{0}$", r"$x_{1}$"])
plt.savefig("examples/double_banana/figures/02_joint.pdf")
plt.clf()

# Generate some fake data
n_conditionals = 10
xs = torch.normal(0.0, 1.0, size=(n_conditionals, 2))
ys = model.param_to_obs(xs)

rs = preconditioner.reference.random(2, 5000)

for i, y_i in enumerate(ys):

    # Draw samples from the associated conditional
    xs_cond, neglogfxs_cond = dirt.eval_cirt(y_i, rs)

    # Run independence MCMC sampler
    neglogfxs_exact = model.neglogpri(xs_cond) + model._negloglik(xs_cond, y_i)
    # xs_mcmc = dt.run_mcmc(xs_cond, neglogfxs_cond, neglogfxs_exact)

    plt.scatter(*xs_cond.T, s=5, zorder=1)
    plt.scatter(xs[i, 0], xs[i, 1], c="k", marker="x", zorder=2)
    plt.xlim(-4.0, 4.0)
    plt.ylim(-4.0, 4.0)
    plt.show()