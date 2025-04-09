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
t_final = 8.0
t_1 = 1.0
dt = 0.2
observation_dt = 0.2
    
ts = np.arange(t_init, t_final + 0.5*dt, dt)
ts_obs = np.arange(t_1, t_final + 0.5*dt, observation_dt)

targets = np.array([[0.0, 0.5], [0.1, 0.4]])

misfit = SpaceTimePointwiseStateObservation(Vh, ts_obs, targets)

prob = HeatSolver(
    mesh=mesh,
    Vh=[Vh, Vh, Vh],
    prior=prior,
    misfit=misfit,
    ts=ts
)

k_true = prob.sample_prior()

hl.nb.plot(k_true)
plt.show()

rel_noise = 0.1
u_true = prob.generate_vector(hl.STATE)
x = [u_true, k_true, None]

t0 = time.time()
prob.solveFwd(x[hl.STATE], x)
t1 = time.time()
print(t1-t0)
    
misfit.observe(x, misfit.d)
MAX = misfit.d.norm("linf", "linf")
noise_std_dev = rel_noise * MAX
hl.parRandom.normal_perturb(noise_std_dev, misfit.d)
misfit.noise_variance = noise_std_dev*noise_std_dev

hl.nb.show_solution(
    Vh, 
    dl.interpolate(prob.u0, prob.Vh[hl.STATE]).vector(), 
    u_true, 
    "Solution",
    times=[0, 0.2, 0.4, 1.0, 4.0, 8.0]
)
plt.show()

# m0 = true_initial_condition.copy()
# _ = hl.modelVerify(problem, m0, is_quadratic=True)