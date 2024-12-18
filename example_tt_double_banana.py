import torch

from deep_tensor.domains import BoundedDomain
from deep_tensor.polynomials import Lagrange1
from deep_tensor.approx_bases import ApproxBases
from deep_tensor.bridging_densities import Tempering1
from deep_tensor.irt.tt_dirt import TTDIRT

from double_banana import DoubleBanana

torch.manual_seed(64)


sigma = 0.3
data = torch.tensor([3.0, 5.0])
model = DoubleBanana(sigma, data)

# Define reference distribution
def reference_dist(u):
    return torch.sqrt(2.0) * torch.erfinv(2*u-1)

# Characterise target distribution
n = 100
xs = torch.linspace(-4.0, 4.0, n)
ys = torch.linspace(-4.0, 4.0, n)
xx, yy = torch.meshgrid(xs, ys, indexing=None)

xx = torch.reshape(xx, (-1, 1))
yy = torch.reshape(yy, (-1, 1))

xts = torch.concatenate([xx, yy], axis=1)
const = 64 / n**2

potential_likelihood, potential_prior = model.potential_dirt(xts)

posterior_density = torch.exp(-potential_likelihood-potential_prior)
rf = torch.exp(-0.5*torch.sum(xts**2, 0))

dim = 2

# Define interpolation bounds
bounds = torch.tensor([-4.0, 4.0])
domain = BoundedDomain(bounds)

# Define interpolation basis
poly = Lagrange1(num_elems=50)
bases = ApproxBases(poly, domain, dim)

# Define bridging measures
bridge = Tempering1()
airt = TTDIRT(model.potential_dirt, bases, bridge=bridge)

print("Done")