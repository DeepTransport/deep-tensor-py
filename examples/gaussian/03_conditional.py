from setup import *


dim_ms = 1
ms = g.mu_post[:dim_ms] + g.cov_post.diag()[:dim_ms].sqrt()
ms = torch.atleast_2d(ms)

rs = reference.random(g.dim - dim_ms, 10_000)
samples, potentials_dirt = dirt.eval_cirt(ms, rs, subset="first")

potentials_true = g.potential_cond(ms, samples)

plot_potentials(potentials_true, potentials_dirt)