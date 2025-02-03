""""""

import time

import deep_tensor as dt

from setup_ou import *


# Estimate the time it takes to carry out the IRT 
# Then the time it takes to compute the Jacobian manually
# Then the time it takes to compute the Jacobian using autodiff
# See how this scales w.r.t. the number of samples that are being processed


poly = "lagrange1"
method = "fixed_rank"

sirt: dt.TTSIRT = sirts[poly][method]

sample_sizes = [10, 50, 100, 500, 1000, 5000, 10_000]

times_irt = []
times_jac = []

for n in sample_sizes:

    zs = torch.rand((n, dim))

    t0 = time.time()
    xs = sirt.eval_irt(zs)[0]
    t1 = time.time()

    times_irt.append(t1-t0)

    t0 = time.time()
    sirt.eval_rt_jac(xs, zs)
    t1 = time.time()

    times_jac.append(t1-t0)

plt.scatter(sample_sizes, times_irt)
# plt.xscale("log")
# plt.yscale("log")
plt.show()
plt.scatter(sample_sizes, times_jac)
# plt.xscale("log")
# plt.yscale("log")
plt.show()