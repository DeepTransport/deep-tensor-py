from setup import *

# rs = reference.random(2, 1000)
# samples, potentials = dirt.eval_irt(rs)
samples = dirt.random(n=10_000)
potentials_dirt = dirt.eval_potential(samples)

potentials_true = g.potential_joint(samples)

print(g.mu_post)
print(samples.mean(dim=0))
print(g.cov_post)
print(torch.cov(samples.T))

plot_potentials(potentials_true, potentials_dirt)

# import corner
# figure = corner.corner(samples.numpy())
# plt.show()
