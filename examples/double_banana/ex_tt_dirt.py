"""TODO: write a docstring for this."""

from matplotlib import pyplot as plt
import torch

import deep_tensor as dt

from examples.double_banana.double_banana import DoubleBanana

plt.style.use("examples/plotstyle.mplstyle")
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
xx, yy = torch.meshgrid(xs, ys, indexing="ij")

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
domain = dt.BoundedDomain(bounds)

# Define interpolation basis
poly = dt.Lagrange1(num_elems=50)
bases = dt.ApproxBases(poly, domain, dim)

# Define bridging measures
bridge = dt.Tempering1()
airt = dt.TTDIRT(model.potential_dirt, bases, bridge=bridge)

n = 100

rxs = torch.linspace(*domain.bounds, n)
rys = torch.linspace(*domain.bounds, n)

xx, yy = torch.meshgrid((rxs, rys), indexing="ij")
xx = torch.reshape(xx, (-1, 1))
yy = torch.reshape(yy, (-1, 1))

rts = torch.concatenate([xx, yy], axis=1)

for k in range(airt.num_layers+1):

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    # Evaluate density
    rf = airt.eval_potential(xts, k)
    rf = torch.exp(-rf)

    neglogliks, neglogpris = model.potential_dirt(xts)

    bf = torch.exp(-neglogliks*airt.bridge.betas[k]-neglogpris)
    bf = bf / (torch.sum(bf) * const)

    axes[0][0].contourf(xs, ys, rf.reshape(n, n).T)
    axes[0][0].set_xlabel("$x_{1}$")
    axes[0][0].set_ylabel("$x_{2}$")
    axes[0][0].set_title(r"$\hat{\pi}$")

    axes[0][1].contourf(xs, ys, bf.reshape(n, n).T)
    axes[0][1].set_xlabel("$x_{1}$")
    axes[0][1].set_ylabel("$x_{2}$")
    axes[0][1].set_title(r"$\pi$")

    fk = airt.irts[k].eval_pdf(rts)
    axes[1][0].contourf(rxs, rys, fk.reshape(n, n).T)
    axes[1][0].set_xlabel("$u_{1}$")
    axes[1][0].set_ylabel("$u_{2}$")
    axes[1][0].set_title("tt")

    if k > 0:
        
        xs_, _ = airt.eval_irt(rts, k)
        neglogliks, neglogpris = model.potential_dirt(xs_)
        logfzs = airt.reference.log_joint_pdf(rts)[0]
        
        bf = torch.exp(-neglogliks*(airt.bridge.betas[k]-airt.bridge.betas[k-1])+logfzs)

        axes[1][1].contourf(rxs, rys, bf.reshape(n, n).T)
        axes[1][1].set_xlabel("$u_{1}$")
        axes[1][1].set_ylabel("$u_{2}$")
        axes[1][1].set_title("a. ratio fun")
    
    else:
        axes[1][1].set_axis_off()

    plt.show()



