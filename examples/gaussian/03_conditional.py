from setup import *


m0_cond = 1.0
ms = torch.tensor([[m0_cond]])
rs = reference.random(1, 1000)
ys, potentials = dirt.eval_cirt(ms, rs, subset="first")

mu_cond = mu_post[1] + cov_post[1][0] * (1 / cov_post[0][0]) * (m0_cond - mu_post[0])
cov_cond = cov_post[1][1] - cov_post[1][0] * (1 / cov_post[0][0]) * cov_post[0][1]

print(ys.mean())
print(mu_cond)
print(ys.std().square())
print(cov_cond)

plt.hist(ys.flatten())
plt.show()

# print(mu_post[0])
# print(samples.mean())
# print(torch.sqrt(cov_post[0, 0]))
# print(torch.std(samples[:, 0].flatten()))

# potentials_true = 0.5 * torch.log(torch.tensor(2.0*torch.pi)) + 0.5 * cov_post[0, 0].log() + 0.5 * ((samples - mu_post[0]).square() / cov_post[0, 0])

# plt.scatter(potentials_true, potentials)
# plt.xlabel("True")
# plt.ylabel("DIRT")
# plt.show()