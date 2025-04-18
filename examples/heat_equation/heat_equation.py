import time

import dolfin as dl
import hippylib as hl
from matplotlib import pyplot as plt
import numpy as np

from solver import SpaceTimePointwiseStateObservation, HeatSolver


mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(3.0, 1.0), 64, 64)


Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

# Generate prior
gamma = 0.5
delta = 0.5

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

noise_std = 1.65e-2
noise_var = noise_std ** 2
misfit = SpaceTimePointwiseStateObservation(
    Vh, 
    ts_obs, 
    targets, 
    noise_variance=noise_var
)

prob = HeatSolver(
    mesh=mesh,
    Vh=[Vh, Vh, Vh],
    prior=prior,
    misfit=misfit,
    ts=ts
)


# for i in range(10):
#     m_true = prob.sample_prior()
#     hl.nb.plot(prob.vec2func(m_true, hl.PARAMETER))
#     plt.show()

m_true = prob.sample_prior()
u_true = prob.generate_vector(hl.STATE)
x = [u_true, m_true, None]

prob.solveFwd(x[hl.STATE], x)

misfit.observe(x, misfit.d)
hl.parRandom.normal_perturb(sigma=noise_std, out=misfit.d)

hl.nb.plot(prob.vec2func(m_true, hl.PARAMETER))
plt.show()

hl.nb.show_solution(
    Vh, 
    prob.u0, 
    u_true, 
    "Solution",
    times=[0, 0.25, 0.5, 1.0, 4.0, 8.0]
)
plt.show()

# prob.x = x
# prob.check_dAuda_fd(x[hl.PARAMETER], x[hl.STATE])

_ = hl.modelVerify(prob, m_true, is_quadratic=False, verbose=True)# , misfit_only=True)
plt.show()

[u, m, p] = prob.generate_vector()
prob.solveFwd(u, [u, m, p])
prob.solveAdj(p, [u, m, p])
mg = prob.generate_vector(hl.PARAMETER)
grad_norm = prob.evalGradientParameter([u, m, p], mg)

# H = hl.ReducedHessian(prob, misfit_only=False) 

# k = 80
# p = 20
# print( "Single Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k, p))
# Omega = hl.MultiVector(x[hl.PARAMETER], k+p)
# hl.parRandom.normal(1., Omega)
# lmbda, V = hl.singlePassG(H, prior.R, prior.Rsolver, Omega, k)

# posterior = hl.GaussianLRPosterior(prior, lmbda, V)

# plt.plot(range(0,k), lmbda, 'b*', range(0,k+1), np.ones(k+1), '-r')
# plt.yscale('log')
# plt.xlabel('number')
# plt.ylabel('eigenvalue')
# plt.show()

# hl.nb.plot_eigenvectors(Vh, V, mytitle="Eigenvector", which=[0,1,2,5,10,20,30,45,60])
# plt.show()

# solver = hl.CGSolverSteihaug()
# solver.set_operator(H)
# solver.set_preconditioner(posterior.Hlr)
# solver.parameters["print_level"] = 1
# solver.parameters["rel_tolerance"] = 1e-6
# solver.solve(m, -mg)
# prob.solveFwd(u, [u, m, p])

m = prior.mean.copy()
solver = hl.ReducedSpaceNewtonCG(prob)
solver.parameters["rel_tolerance"] = 1e-6
solver.parameters["abs_tolerance"] = 1e-12
solver.parameters["max_iter"]      = 25
solver.parameters["GN_iter"] = 5
solver.parameters["globalization"] = "LS"
solver.parameters["LS"]["c_armijo"] = 1e-4


x = solver.solve([None, m, None])


total_cost, reg_cost, misfit_cost = prob.cost([u, m, p])
print( "Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(total_cost, reg_cost, misfit_cost) )

total_cost, reg_cost, misfit_cost = prob.cost([u_true, m_true, p])
print( "Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(total_cost, reg_cost, misfit_cost) )

hl.nb.show_solution(
    Vh, 
    prob.u0, 
    u, 
    "Solution",
    times=[0, 0.25, 0.5, 1.0, 4.0, 8.0]
)
plt.show()

hl.nb.show_solution(
    Vh, 
    prob.u0, 
    u_true, 
    "Solution",
    times=[0, 0.25, 0.5, 1.0, 4.0, 8.0]
)
plt.show()

hl.nb.plot(prob.vec2func(m_true, hl.PARAMETER))
plt.show()

hl.nb.plot(prob.vec2func(m, hl.PARAMETER))
plt.show()

pass