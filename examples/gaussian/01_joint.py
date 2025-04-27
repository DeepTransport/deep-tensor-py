from setup import *

# rs = reference.random(2, 1000)
# samples, potentials = dirt.eval_irt(rs)
samples = dirt.random(n=1000)
potentials = dirt.eval_potential(samples)

potentials_true = torch.log(torch.tensor(2.0*torch.pi)) + 0.5 * torch.logdet(cov_post) + 0.5 * ((samples - mu_post) @ R_post).square().sum(dim=1)

print(mu_post)
print(samples.mean(dim=0))
print(cov_post)
print(torch.cov(samples.T))

# plt.scatter(*samples.T)
# plt.show()

plt.scatter(potentials_true, potentials)
plt.xlabel("True")
plt.ylabel("DIRT")
plt.show()