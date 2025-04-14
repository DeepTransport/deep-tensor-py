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
    return mu_pri + xs @ L_pri.T

def Q_inv(ms: Tensor) -> Tensor:
    return (ms - mu_pri) @ R_pri.T

def neglogabsdet_Q_inv(ms: Tensor):
    return torch.full((ms.shape[0],), -det_R)

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

poly = dt.Fourier(order=40)
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

samples = dirt.random(n=1000)
potentials = dirt.eval_potential(samples)

mu_post = mu_pri + cov_pri @ G.T @ linalg.inv(G @ cov_pri @ G.T + cov_e) @ (y_obs - G @ mu_pri)
cov_post = cov_pri - cov_pri @ G.T @ linalg.inv(G @ cov_pri @ G.T + cov_e) @ G @ cov_pri
L_post = linalg.cholesky(torch.linalg.inv(cov_post))

potentials_true = torch.log(torch.tensor(2.0*torch.pi)) + 0.5 * torch.logdet(cov_post) + 0.5 * ((samples - mu_post) @ L_post.T).square().sum(dim=1)

print(mu_post)
print(samples.mean(dim=0))
print(cov_post)
print(torch.cov(samples.T))

# plt.scatter(*samples.T)
# plt.show()

plt.scatter(potentials_true, potentials)
plt.xlabel("True")
plt.ylabel("DIRT")
plt.show()