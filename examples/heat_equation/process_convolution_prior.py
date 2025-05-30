from matplotlib import pyplot as plt
import torch


# Define grid on which to evaluate process
x0s = torch.linspace(0.0, 3.0, 100)
x1s = torch.linspace(0.0, 1.0, 100)
xs = torch.tensor([[x0, x1] for x0 in x0s for x1 in x1s])

# Define locations of Gaussian white noise
s0s = 3 * torch.linspace(1/14, 13/14, 7)
s1s = torch.linspace(1/6, 5/6, 3)
ss = torch.tensor([[s0, s1] for s0 in s0s for s1 in s1s])

n, d = xs.shape
m = ss.shape[0]

xs_exp = xs.unsqueeze(1).expand(n, m, d)
ss_exp = ss.unsqueeze(0).expand(n, m, d)

dist = torch.pow(xs_exp-ss_exp, 2).sum(dim=2)
dist = torch.exp(-4 * dist)

xis = torch.normal(0.0, 1.0, size=(m,))

field = dist @ xis

field = field.reshape(100, 100)


plt.pcolormesh(x0s, x1s, field.T)
plt.scatter(*ss.T)
plt.gca().set_aspect('equal', 'box')
plt.colorbar()
plt.show()