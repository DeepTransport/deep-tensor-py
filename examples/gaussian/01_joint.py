from setup import *

# rs = reference.random(2, 1000)
# samples, potentials = dirt.eval_irt(rs)
samples = dirt.random(n=10_000)

potentials_true = g.potential_joint(samples)
potentials_dirt = dirt.eval_potential(samples)

# Verify that RT and IRT are inverses of one another
xs = dirt.eval_rt(samples)[0]
ms = dirt.eval_irt(xs)[0]
print(torch.linalg.norm(samples-ms))

print(g.mu_post)
print(samples.mean(dim=0))
print(g.cov_post)
print(torch.cov(samples.T))

plot_potentials(potentials_true, potentials_dirt)

# import corner
# figure = corner.corner(samples.numpy())
# plt.show()
