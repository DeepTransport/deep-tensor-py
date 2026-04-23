import math

import torch 
from torch import Tensor

import deep_tensor as dt

torch.manual_seed(0)


def cov2cor(cov: Tensor) -> Tensor:
    Dinv = torch.diag(1.0 / cov.diag().sqrt())
    cor = Dinv @ cov @ Dinv
    return cor


def generate_random_cov(n):
    """Generates a random covariance matrix."""
    # Generate an orthogonal basis
    A = torch.randn((n, n))
    Q = torch.linalg.qr(A)[0]
    # Generate a set of singular values
    sigmas = torch.abs(torch.randn(n)) + 1e-8
    cov = (Q.T * sigmas) @ Q
    return cov
    

def generate_random_cor(n):
    """Generates a random correlation matrix."""
    cov = generate_random_cov(n)
    return cov2cor(cov)

dim = 4

m = torch.zeros((dim, 1))
C = generate_random_cor(dim)
C_inv = torch.linalg.inv(C)
L_inv = torch.linalg.cholesky(C_inv)

print(C)

def neglogpri(xs: Tensor) -> Tensor:
    return 0.5 * (L_inv.T @ (xs.T - m)).square().sum(dim=0)

def negloglik(xs: Tensor) -> Tensor:
    return torch.zeros((xs.shape[0],))


preconditioner = dt.IdentityMapping(dim=dim)
bases = dt.Fourier(order=15)

bridge = dt.Tempering(betas=torch.tensor([0.0001, 0.001, 0.01, 0.1, 1.0]))

dirt = dt.DIRT(
    negloglik, 
    neglogpri, 
    preconditioner=preconditioner, 
    bases=bases,
    bridge=bridge
)

rs = dirt.reference.random(d=dirt.dim, n=100_000)
xs, potentials_dirt = dirt.eval_irt(rs)

neglognorm = 0.5 * dirt.dim * math.log(2.0 * math.pi)
potentials_true = neglognorm + neglogpri(xs)

from tests.test_dirt.plotting import plot_potentials

plot_potentials(potentials_dirt, potentials_true)

# print(C)

#print(xs.mean(dim=0))
#print(xs.T.cov())