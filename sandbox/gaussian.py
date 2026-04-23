import math
import torch 
from torch import Tensor 

import deep_tensor as dt
from deep_tensor.debiasing.mcmc.pcn import pCNKernel
from deep_tensor.debiasing.mcmc.mcmc import MCMC


def neglogtarget(xs: Tensor) -> Tensor:
    """The unit normal density."""
    return 0.5 * xs.square().sum(dim=1) + 0.5 * xs.shape[1] * math.log(2.0*math.pi)


dim = 10

target_func = dt.TargetFunc(neglogtarget)

reference = dt.GaussianReference(domain=dt.BoundedDomain([-4.0, 4.0]))
preconditioner = dt.IdentityMapping(dim, reference)

basis = dt.Fourier(order=10)
bases = dt.ApproxBases(basis, dim)
tt_options = dt.TTOptions(tt_method="fixed_rank", init_rank=1)
tt = dt.TT(tt_options)
ftt = dt.FTT(bases, tt)

bridge = dt.SingleLayer()

dirt = dt.DIRT(target_func, preconditioner, ftt, bridge)

rs = dirt.reference.random(n=10_000, d=dim)
xs, neglogfxs_dirt = dirt.eval_irt(rs)
neglogfxs_true = neglogtarget(xs)
print(neglogfxs_dirt)
print(neglogfxs_true)
print(dt.run_independence_sampler(xs, neglogfxs_dirt, neglogfxs_true).acceptance_rates)

num_chains = 100
r0s = torch.zeros((num_chains, dim))

kernel = pCNKernel(neglogtarget, dirt, dt=2.0)
mcmc = MCMC(kernel, r0s, num_steps=1000)
res = mcmc.run()

print(res.acceptance_rates)
print(res.iacts)