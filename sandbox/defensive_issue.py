from matplotlib import pyplot as plt
import torch 
from torch import Tensor

from examples.plotting import pairplot

import deep_tensor as dt


torch.manual_seed(0)


dim = 20

# mu = torch.full((dim,), 3.0)
# sd = 0.1

mu = torch.full((dim,), 1.0)
sd = 1.0

def neglogfx(xs: Tensor) -> Tensor:
    return (1 / (2.0 * sd**2)) * (xs - mu).square().sum(dim=1) - 26.0


target_func = dt.TargetFunc(neglogfx)

domain = dt.BoundedDomain([-4.0, 4.0])
reference = dt.GaussianReference(domain)
preconditioner = dt.IdentityMapping(dim, reference)

basis = dt.Lagrange1(num_elems=30)
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

num_samples = 10_000
rs = dirt.reference.random(n=num_samples, d=dim)
xs, neglogfxs_dirt = dirt.eval_irt(rs)
neglogfxs_exact = neglogfx(xs)

res = dt.run_importance_sampling(neglogfxs_dirt, neglogfxs_exact)

import math
print(math.exp(dirt.log_z))
for k in dirt.sirts:
    print(f"{dirt.sirts[k].z = }")
    print(f"{dirt.sirts[k].z_func = }")

# print(dirt.log_z)
# z = (2.0*torch.pi)**(-dim/2.0)
# import math
# print(math.exp(dirt.log_z))
# print(z)

print(f"{res.ess = }")

# pairplot(xs[:, :4])
# plt.show()

# pairplot(xs[:, -4:])
# plt.show()