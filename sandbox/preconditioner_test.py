import torch

import deep_tensor as dt 

torch.manual_seed(0)

dim = 10

# preconditioner = dt.IdentityMapping(dim)

# A = torch.randn((dim, dim))
# b = torch.randn((dim,))
# preconditioner = dt.AffineMapping(A, b)

mu = torch.ones((dim,))
cov = 2.0*torch.eye(dim)
preconditioner = dt.GaussianMapping(mu, cov, diag=True)

us = torch.randn((5, dim))

xs, neglogdets, dxdus = preconditioner.grad_Q(us[:, :2], subset="last")

print(dxdus[:, 2, :])