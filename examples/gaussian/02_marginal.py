from setup import *


zs = reference.random(1, 1000)
samples, potentials = dirt.eval_irt(zs, subset="first")

print(mu_post[0])
print(samples.mean())
print(torch.sqrt(cov_post[0, 0]))
print(torch.std(samples[:, 0].flatten()))

potentials_true = 0.5 * torch.log(torch.tensor(2.0*torch.pi)) + 0.5 * cov_post[0, 0].log() + 0.5 * ((samples - mu_post[0]).square() / cov_post[0, 0])

plt.scatter(potentials_true, potentials)
plt.xlabel("True")
plt.ylabel("DIRT")
plt.show()