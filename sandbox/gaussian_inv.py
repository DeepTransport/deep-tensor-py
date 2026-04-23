"""Eventually, this will be a unit test with a Gaussian density."""

import math

import torch 
from torch import Tensor

import deep_tensor as dt

torch.manual_seed(0)


def generate_random_mean(n):
    return torch.randn((n,))


def generate_random_cov(n):
    """Generates a random covariance matrix."""
    # Generate an orthogonal basis
    A = torch.randn((n, n))
    Q = torch.linalg.qr(A)[0]
    # Generate a set of singular values
    sigmas = torch.abs(torch.randn(n)) + 1e-8
    cov = (Q.T * sigmas) @ Q
    return cov


def generate_random_G(num_param, num_data):
    G = torch.randn((num_data, num_param))
    return G


class LinearGaussianProblem():

    def __init__(
        self, 
        mean_pri: Tensor, 
        cov_pri: Tensor, 
        G: Tensor,
        y_obs: Tensor,
        sd_noise: float
    ):
        
        if y_obs.ndim == 1:
            y_obs = y_obs[:, None]

        if mean_pri.ndim == 1:
            mean_pri = mean_pri[:, None]

        self.m_pri = mean_pri 
        self.Cm = cov_pri 
        self.Cm_inv = torch.linalg.inv(self.Cm)
        self.Lm_inv = torch.linalg.cholesky(self.Cm_inv)
        
        self.G = G 
        self.y_obs = y_obs

        self.n = self.m_pri.numel()
        self.m = self.y_obs.numel()
        
        self.sd_noise = sd_noise
        self.Ce = sd_noise ** 2 * torch.eye(y_obs.numel())
        self.Ce_inv = torch.linalg.inv(self.Ce)
        self.Le_inv = torch.linalg.cholesky(self.Ce_inv)

        # Compute properties of posterior
        self.C_post = torch.linalg.inv(self.G.T @ self.Ce_inv @ self.G + self.Cm_inv)
        self.m_post = self.C_post @ (self.G.T @ self.Ce_inv @ self.y_obs + self.Cm_inv @ self.m_pri)

        self.C_post_inv = torch.linalg.inv(self.C_post)
        self.L_post_inv = torch.linalg.cholesky(self.C_post_inv)

        self.neglogpostnorm = (0.5 * self.n * math.log(2.0 * math.pi) 
                               + 0.5 * torch.logdet(self.C_post))

        return
    
    def negloglik(self, xs: Tensor) -> Tensor:
        Gs = self.G @ xs.T
        misfit = (self.Le_inv.T @ (Gs - self.y_obs)).square().sum(dim=0)
        neglogliks = 0.5 * misfit
        return neglogliks

    def neglogpri(self, xs: Tensor) -> Tensor:
        misfit = (self.Lm_inv.T @ (xs.T - self.m_pri)).square().sum(dim=0)
        neglogpris = 0.5 * misfit
        return neglogpris
    
    def neglogpost(self, xs: Tensor) -> Tensor:
        misfit = (self.L_post_inv.T @ (xs.T - self.m_post)).square().sum(dim=0)
        neglogposts = self.neglogpostnorm + 0.5 * misfit
        return neglogposts


n = 4
m = 2

mean_pri = generate_random_mean(n)
cov_pri = generate_random_cov(n)
mean_pri = torch.zeros((n,))
cov_pri = torch.eye(n)

G = generate_random_G(n, m)

print(G)

# mean_pri = 1.0 + torch.tensor([0.0, 0.0])
# cov_pri = 2 * torch.tensor([[1.0, 0.5], 
#                             [0.5, 1.0]])

# G = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
sd_noise = 1.0

param_true = torch.distributions.MultivariateNormal(mean_pri, cov_pri).rsample()
noise = sd_noise * torch.randn((m,))

y_obs = G @ param_true + noise

prob = LinearGaussianProblem(mean_pri, cov_pri, G, y_obs, sd_noise)

dirt = dt.DIRT(
    prob.negloglik, 
    prob.neglogpri, 
    preconditioner=dt.IdentityMapping(dim=n), 
    bases=dt.Lagrange1(num_elems=30)
)

samples = dirt.random(n=100_000)

mean_dirt = samples.mean(dim=0)
cov_dirt = samples.T.cov()

potentials_dirt = dirt.eval_potential(samples)
potentials_true = prob.neglogpost(samples)

import matplotlib.pyplot as plt 
plt.scatter(potentials_dirt, potentials_true, s=4)
plt.show()