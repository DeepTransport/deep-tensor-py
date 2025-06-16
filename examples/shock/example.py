#%% Example: Shock absorber

import os
from matplotlib import pyplot as plt
import torch
import deep_tensor as dt

from preconditioner import construct_preconditioner
from examples.plotting import pairplot


torch.manual_seed(0)
plt.style.use(os.sep.join(["examples", "plotstyle.mplstyle"]))


# Define failure distances (km)
failure_dists = torch.tensor([
    6700,  6950,  7820,  8790,  9120,
    9660,  9820,  11310, 11690, 11850, 
    11880, 12140, 12200, 12870, 13150, 
    13330, 13470, 14040, 14300, 17520,
    17540, 17890, 18420, 18960, 18980,
    19410, 20100, 20100, 20150, 20320, 
    20900, 22700, 23490, 26510, 27410, 
    27490, 27890, 28100
])

# Define whether or not each observation is right-censored
censored = torch.tensor([
    False, True,  True,  True,  False, 
    True,  True,  True,  True,  True, 
    True,  True,  False, True,  False, 
    True,  True,  True,  False, False, 
    True,  True,  True,  True,  True, 
    True,  False, True,  True,  True, 
    False, False, True,  False, True, 
    False, True,  True
])

# Generate covariates
D = 2
xs = torch.randn((failure_dists.numel(), D)) / D

# Define prior coefficients
alpha = 6.8757
gamma = 2.2932

ms = torch.zeros((D+1,))
ms[0] = torch.tensor(30796).log()
sds = torch.ones((D+1,))
sds[0] = torch.tensor(0.1563).sqrt()

#%% DIRT construction

def negloglik(params: torch.Tensor) -> torch.Tensor:

    bs, t2s = params[:, :-1], params[:, -1:]

    # Compute theta_1 values
    bxs = torch.sum(bs[:, 1:, None] * xs.T[None, :, :], dim=1)
    t1s = torch.exp(bs[:, :1] + bxs)

    # Add contribution of uncensored failure times
    neglogliks = (
        -torch.log(t2s / t1s[:, ~censored]) 
        - (t2s - 1.0) * torch.log(failure_dists[~censored] / t1s[:, ~censored]) 
        + (failure_dists[~censored] / t1s[:, ~censored]) ** t2s
    ).sum(dim=1)
    
    # Add contribution of censored failure times
    neglogliks += (
        (failure_dists[censored] / t1s[:, censored]) ** t2s
    ).sum(dim=1)
    
    neglogliks = neglogliks - 144.0  # Numerical stability
    return neglogliks


def neglogpri(params: torch.Tensor) -> torch.Tensor:

    bs, t2s = params[:, :-1], params[:, -1:]
    
    neglogpris = (
        - (0.5*D + alpha) * t2s.log().flatten()
        + gamma * t2s.flatten()
        + 0.5 * torch.sum(t2s * (bs-ms)**2 / sds**2, dim=1)
    )
    
    return neglogpris


# Define bounds for parameters
bounds = torch.zeros((D+2, 2))
bounds[:-1] = torch.vstack((ms - 3.0*sds, ms + 3.0*sds)).T 
bounds[-1] = torch.tensor([1e-8, 13.0])

# Construct mapping from Gaussian reference to prior
dim = D + 2
reference = dt.GaussianReference()
preconditioner = construct_preconditioner(
    reference, bounds, 
    alpha, gamma, 
    ms, sds, dim
)
# preconditioner = dt.UniformMapping(bounds)

bases = dt.Lagrange1(num_elems=20)
bridge = dt.SingleLayer()
tt_options = dt.TTOptions(verbose=2, max_als=2, init_rank=10, max_rank=14)

dirt = dt.DIRT(
    negloglik=negloglik,
    neglogpri=neglogpri,
    preconditioner=preconditioner,
    bases=bases,
    bridge=bridge,
    tt_options=tt_options
)

#%% Debiasing

# Generate a set of samples from the DIRT approximation to the posterior
rs = reference.random(d=dirt.dim, n=50_000)
samples_dirt, potentials_dirt = dirt.eval_irt(rs)

# Run an independence MCMC sampler
potentials_true = negloglik(samples_dirt) + neglogpri(samples_dirt)
res = dt.run_independence_sampler(samples_dirt, potentials_dirt, potentials_true)

print(f"Acceptance rate: {res.acceptance_rate:.2f}")
print(f"Mean IACT: {res.iacts.mean():.2f}")
print(f"Max IACT: {res.iacts.max():.2f}")

# Thin the chain
samples_post = res.xs[::5, :]

# Generate a set of samples from the prior
samples_prior = preconditioner.Q(rs[::5, :])

labels = [r"$\beta_{0}$", r"$\beta_{1}$", r"$\beta_{2}$", r"$\theta_{2}$"]
pairplot(
    samples_post, 
    samples_prior,
    labels=labels,
    x_label="Posterior",
    y_label="Prior"
)

res = dt.run_importance_sampling(potentials_dirt, potentials_true)
ess_ratio = res.ess / potentials_dirt.numel()
print(f"ESS ratio: {ess_ratio:.2f}")