from setup import *


dim_r = 3

rs = reference.random(dim_r, 1000)
ms, potentials_dirt = dirt.eval_irt(rs, subset="first")

print(g.mu_post[:dim_r])
print(ms.mean(dim=0))
print(g.cov_post[:dim_r, :dim_r])
print(torch.cov(ms.T))

potentials_true = g.potential_marg(ms)

plot_potentials(potentials_true, potentials_dirt)