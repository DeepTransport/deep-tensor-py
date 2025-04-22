"""Builds a DIRT approximation to a 2D Gaussian with an arbitrary mean 
and covariance matrix.
"""

from matplotlib import pyplot as plt
import torch
from torch import linalg
from torch import Tensor
from torch.distributions import MultivariateNormal

import deep_tensor as dt

plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)

dim = 2

G = torch.tensor([[2.0, 0.0],
                  [0.0, 2.0]])

mu_pri = torch.tensor([1.0, 0.0])
cov_pri = torch.tensor([[2.0, 1.0], 
                        [1.0, 2.0]])
L_pri = linalg.cholesky(cov_pri)
R_pri = linalg.inv(L_pri)
det_R = torch.logdet(R_pri)

def Q(xs: Tensor) -> Tensor:
    d_xs = xs.shape[1]
    return mu_pri[:d_xs] + xs @ L_pri[:d_xs, :d_xs].T

def Q_inv(ms: Tensor) -> Tensor:
    d_ms = ms.shape[1]
    return (ms - mu_pri[:d_ms]) @ R_pri[:d_ms, :d_ms].T

def neglogabsdet_Q_inv(ms: Tensor):
    n_ms, d_ms = ms.shape
    return torch.full((n_ms, ), -R_pri[:d_ms, :d_ms].diag().log().sum())

bounds = torch.tensor([-4.0, 4.0])
domain = dt.BoundedDomain(bounds=bounds)
reference = dt.GaussianReference(domain=domain)

prior = dt.PriorTransformation(
    reference, 
    Q, 
    Q_inv, 
    neglogabsdet_Q_inv, 
    dim
)

mu_e = torch.tensor([0.0, 0.0])
cov_e = 1.0 ** 2 * torch.tensor([[1.0, 0.0],
                                 [0.0, 1.0]])
R_e = linalg.inv(linalg.cholesky(cov_e))

# Generate some error terms
error_dist = MultivariateNormal(mu_e, cov_e)
error = error_dist.sample()

# Generate some observations
m_true = torch.tensor([1.0, 1.0])
y_obs = G @ m_true + error

def negloglik(ms: Tensor) -> Tensor:
    return 0.5 * ((ms @ G.T - y_obs) @ R_e).square().sum(dim=1)

# Define true posterior
mu_post = mu_pri + cov_pri @ G.T @ linalg.inv(G @ cov_pri @ G.T + cov_e) @ (y_obs - G @ mu_pri)
cov_post = cov_pri - cov_pri @ G.T @ linalg.inv(G @ cov_pri @ G.T + cov_e) @ G @ cov_pri
L_post = linalg.cholesky(torch.linalg.inv(cov_post))

poly = dt.Fourier(order=20)
# tt_options = dt.TTOptions(max_cross=4)
# bridge = dt.SingleLayer()
bridge = dt.Tempering()

dirt = dt.TTDIRT(
    negloglik, 
    prior,
    poly, 
    bridge=bridge
    # sirt_options=tt_options
    # bridge
)