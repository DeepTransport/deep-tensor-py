from matplotlib import pyplot as plt
import torch 
from torch import Tensor

from examples.plotting import pairplot

import deep_tensor as dt


torch.manual_seed(0)


dim = 3

# mu = torch.full((dim,), 3.0)
# sd = 0.1

mu = torch.full((dim,), 0.0)
cov = torch.tensor([[1.5, 0.5, 0.5], 
                    [0.5, 1.0, 0.5],
                    [0.5, 0.5, 1.0]])

R = torch.linalg.cholesky(cov)
L = torch.linalg.inv(R)

def neglogfx(xs: Tensor) -> Tensor:
    return ((xs - mu) @ L.T).square().sum(dim=1)


target_func = dt.TargetFunc(neglogfx)

domain = dt.BoundedDomain([-4.0, 4.0])
reference = dt.GaussianReference(domain)
preconditioner = dt.IdentityMapping(dim, reference)

basis = dt.Fourier(order=10)
bases = dt.ApproxBases(basis, dim)
tt_options = dt.TTOptions(tt_method="fixed_rank", init_rank=3, verbose=2)
tt = dt.TT(tt_options)
ftt = dt.FTT(bases, tt)

bridge = dt.SingleLayer()
# betas = 10 ** torch.linspace(-4.0, 0.0, 5)
# bridge = dt.Tempering(betas)

#dirt_options = dt.DIRTOptions(defensive=0.0)
dirt_options = dt.DIRTOptions(defensive=1e-10)
dirt = dt.DIRT(target_func, preconditioner, ftt, bridge, dirt_options)

nx = 200
xs_k = torch.linspace(-4.0, 4.0, nx)
dx = xs_k[1] - xs_k[0]

neglogfxs_k = dirt.eval_marginal(xs_k, k=0)

fxs_k = torch.exp(-neglogfxs_k)

from matplotlib import pyplot as plt 
plt.plot(xs_k, fxs_k)
plt.show()

print("done")

# num_samples = 10_000
# rs = dirt.reference.random(n=num_samples, d=dim)
# xs, neglogfxs_dirt = dirt.eval_irt(rs)
# neglogfxs_exact = neglogfx(xs)

