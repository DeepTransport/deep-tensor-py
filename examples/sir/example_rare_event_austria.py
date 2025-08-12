from typing import Tuple

import numpy as np
import torch 

import deep_tensor as dt

from examples.sir import SIRCompartmentModel


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


num_compartments = 9
dim = 2 * num_compartments

adjacency_mat = np.load("examples/sir/data/austria_adjacency.npy")

# Specify initial condition
S0 = np.full((num_compartments,), 100.0)
I0 = np.zeros((num_compartments,))
R0 = np.zeros((num_compartments,))
S0[0] = 99.0
I0[0] = 1.0

# Specify time horizon and evaluation times
t1 = 5.0
t_eval = np.linspace(0, 5, 13*12+1)
inds_obs = np.arange(13, 13*12+1, 13)

# Specify properties of likelihood
std_noise = 1.0

# Define model
model = SIRCompartmentModel(adjacency_mat, S0, I0, R0, t1, t_eval, inds_obs)

# Generate synthetic observations
true_param = torch.tensor([[0.1, 1.0] * num_compartments])
ys_true = model.get_obs(model.solve(true_param))
ys_obs = ys_true # + std_noise * torch.randn_like(ys_true)


def neglogpost(xs: torch.Tensor) -> torch.Tensor:
    # Note: prior is uniform
    ys = model.get_obs(model.solve(xs))
    neglogliks = (ys-ys_obs).square().sum(dim=1) / (2*std_noise**2)
    return neglogliks


def rare_event_func(xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    ys_full = model.solve(xs)
    ys = model.get_obs(ys_full)

    neglogliks = (ys-ys_obs).square().sum(dim=1) / (2*std_noise**2)
    response = model.response_func(ys_full)

    return neglogliks, response


# Define rare event threshold
I_max = 69.0

# Define reference density
domain = dt.BoundedDomain(bounds=torch.tensor([-3.0, 3.0]))
reference = dt.GaussianReference(domain)

# Define preconditioner
bounds = torch.tensor([[0.0, 2.0]]).tile((dim, 1))
preconditioner = dt.UniformMapping(bounds=bounds)

bases = dt.Lagrange1(num_elems=16)

tt_options = dt.TTOptions(init_rank=1, max_rank=5, max_als=3, local_tol=0.0, cdf_tol=1e-10, verbose=2)

# Construct the numerator of the ratio estimator (i.e., the DIRT 
# approximation to the optimal biasing density)

rare_event = dt.RareEventFunc(rare_event_func, threshold=I_max)

betas = 10 ** torch.linspace(-5.0, 0.0, 16)
gamma_prime = 1e4 / I_max
gammas = betas * gamma_prime
bridge = dt.SigmoidSmoothing(gammas, betas)

numerator = dt.DIRT(
    rare_event, 
    preconditioner, 
    bases, 
    bridge, 
    tt_options=tt_options
)

n_samples = 10_000

rs = numerator.reference.random(dim, n_samples)
xs, neglogfxs_dirt = numerator.eval_irt(rs)
neglogfxs_exact = rare_event(xs)

Q_is = dt.run_importance_sampling(neglogfxs_dirt, neglogfxs_exact)
Q_hat = Q_is.log_weights.exp().mean()

# Construct the denominator of the ratio estimator (i.e., the DIRT 
# approximation to the posterior)

posterior = dt.TargetFunc(neglogpost)

betas = 10 ** torch.linspace(-5.0, 0.0, 16)
betas = betas.tolist()
bridge = dt.Tempering(betas)

denominator = dt.DIRT(
    posterior, 
    preconditioner, 
    bases, 
    bridge, 
    tt_options=tt_options
)

rs = denominator.reference.random(dim, n_samples)
xs, neglogfxs_dirt = denominator.eval_irt(rs)
neglogfxs_exact = posterior(xs)

Z_is = dt.run_importance_sampling(neglogfxs_dirt, neglogfxs_exact)
Z_hat = Z_is.log_weights.exp().mean()

R_hat = Q_hat / Z_hat

print(R_hat)