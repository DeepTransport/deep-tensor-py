import time

import dolfin as dl
import hippylib as hl
from matplotlib import pyplot as plt
import numpy as np

from solver import SpaceTimePointwiseStateObservation, HeatSolver


mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(3.0, 1.0), 64, 64)


Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

# Generate prior
gamma = 0.1
delta = 8.0

prior_mean = dl.interpolate(dl.Constant(-5.0), Vh).vector()
prior = hl.BiLaplacianPrior(
    Vh, 
    gamma, 
    delta, 
    mean=prior_mean,
    robin_bc=True
)

t_init = 0.0
t_final = 10.0
t_1 = 1.0
dt = 0.25
dt_obs = 1.0
    
ts = np.arange(t_init, t_final+0.5*dt, dt)
ts_obs = np.arange(t_1, t_final+0.5*dt, dt_obs)

targets = np.array([
    [0.4, 0.4], 
    [0.4, 0.6],
    [0.6, 0.4],
    [0.6, 0.6],
    [2.4, 0.4],
    [2.4, 0.5],
    [2.4, 0.6],
    [2.5, 0.4],
    [2.5, 0.5],
    [2.5, 0.6],
    [2.6, 0.4],
    [2.6, 0.5],
    [2.6, 0.6]
])

misfit = SpaceTimePointwiseStateObservation(Vh, ts_obs, targets)

prob = HeatSolver(
    mesh=mesh,
    Vh=[Vh, Vh, Vh],
    prior=prior,
    misfit=misfit,
    ts=ts
)

k_true = prob.sample_prior()

# hl.nb.plot(k_true)
# plt.show()

u_true = prob.generate_vector(hl.STATE)
x = [u_true, k_true, None]

t0 = time.time()
prob.solveFwd(x[hl.STATE], x)
t1 = time.time()
print(t1 - t0)
    
misfit.observe(x, misfit.d)
noise_std_dev = 1.65e-2
hl.parRandom.normal_perturb(sigma=noise_std_dev, out=misfit.d)
misfit.noise_variance = noise_std_dev ** 2

# hl.nb.show_solution(
#     Vh, 
#     dl.interpolate(prob.u0, prob.Vh[hl.STATE]).vector(), 
#     u_true, 
#     "Solution",
#     times=[0, 0.25, 0.5, 1.0, 4.0, 8.0]
# )
# plt.show()

_ = hl.modelVerify(prob, k_true, is_quadratic=True)

# [u, m, p] = prob.generate_vector()
# prob.solveFwd(u, [u, m, p])
# prob.solveAdj(p, [u, m, p])
# mg = prob.generate_vector(hl.PARAMETER)
# grad_norm = prob.evalGradientParameter([u, m, p], mg)