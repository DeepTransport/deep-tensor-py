"""
Some things to do:
    - Define proper problem setup.
    - Define process convolution prior.
    - Build ROM?
"""

import time

import dolfin as dl
import hippylib as hl
from matplotlib import pyplot as plt
import numpy as np

from solver import *


mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(3.0, 1.0), 64, 64)

Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

# Generate prior
gamma = 0.1
delta = 0.5

# prior_mean = dl.interpolate(dl.Constant(-4.0), Vh).vector()
# prior = hl.BiLaplacianPrior(
#     Vh, 
#     gamma, 
#     delta, 
#     mean=prior_mean,
#     robin_bc=True
# )

# Define locations of Gaussian white noise
s0s = 3 * torch.linspace(1/14, 13/14, 7)
s1s = torch.linspace(1/6, 5/6, 3)
ss = torch.tensor([[s0, s1] for s0 in s0s for s1 in s1s])

prior = ProcessConvolutionPrior(
    mesh, 
    Vh, 
    ss, 
    mu=-4.0, 
    r=4.0
)

dt = 0.25
dt_obs = 1.0
t_init = dt
t_final = 10.0
t_1 = dt_obs
    
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

# targets = np.array([
#     [x, y] 
#     for x in np.linspace(0.0, 3.0, 20)
#     for y in np.linspace(0.0, 1.0, 20)
# ])

std_error = 1.65e-3
var_error = std_error ** 2
misfit = SpaceTimePointwiseStateObservation(Vh, ts_obs, targets, var_error)

prob = HeatSolver(
    mesh=mesh,
    Vh=[Vh, Vh, Vh],
    prior=prior,
    misfit=misfit,
    ts=ts
)


for i in range(20):

    m_true = prob.sample_prior()
    u_true = prob.generate_vector(hl.STATE)
    x = [u_true, m_true, None]

    prob.solveFwd(x[hl.STATE], x)

# prob.check_jacobian_fd(x)
jac = prob.eval_jacobian(x)

# m_true = prob.sample_prior()
# u_true = prob.generate_vector(hl.STATE)
# x = [u_true, m_true, None]

# prob.solveFwd(x[hl.STATE], x)

misfit.observe(x)
hl.parRandom.normal_perturb(sigma=std_error, out=misfit.d)

# hl.nb.plot(prob.vec2func(m_true, hl.PARAMETER))
# plt.show()

# hl.nb.show_solution(
#     Vh, 
#     prob.u0, 
#     u_true, 
#     "Solution",
#     times=[0, 0.25, 0.5, 1.0, 4.0, 8.0]
# )
# plt.show()

prob.x = x
# prob.check_dAuda_fd(x[hl.PARAMETER], x[hl.STATE])
# prob.check_gradient_fd(m_true)
# prob.check_gradient_fd(prior.mean)
# prob.check_forward(x)
# prob.check_adjoint(x)

_ = hl.modelVerify(prob, prior.mean, is_quadratic=False, verbose=True)# , misfit_only=True)
plt.show()

[u, m, p] = prob.generate_vector()
# m = prior.mean.copy()
m = prob.sample_prior()
prob.solveFwd(u, [u, m, p])
prob.solveAdj(p, [u, m, p])
mg = prob.generate_vector(hl.PARAMETER)
grad_norm = prob.evalGradientParameter([u, m, p], mg)

# hl.nb.plot_eigenvectors(Vh, V, mytitle="Eigenvector", which=[0,1,2,5,10,20,30,45,60])
# plt.show()

# solver = hl.CGSolverSteihaug()
# solver.set_operator(H)
# solver.set_preconditioner(posterior.Hlr)
# solver.parameters["print_level"] = 1
# solver.parameters["rel_tolerance"] = 1e-6
# solver.solve(m, -mg)
# prob.solveFwd(u, [u, m, p])

# m = prior.mean.copy()
parameters = hl.ReducedSpaceNewtonCG_ParameterList()
# parameters["rel_tolerance"] = 1e-9
# parameters["abs_tolerance"] = 1e-12
# parameters["max_iter"]      = 25
# parameters["globalization"] = "LS"
parameters["GN_iter"] = 25

# params = hl.ReducedSpaceNewtonCG_ParameterList(GN_iter=20)
solver = hl.ReducedSpaceNewtonCG(prob, parameters)

# solver = hl.SteepestDescent(prob)

x = solver.solve([u, m, p])
u, m, p = x

total_cost, reg_cost, misfit_cost = prob.cost([u, m, p])
print( "Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(total_cost, reg_cost, misfit_cost) )

total_cost, reg_cost, misfit_cost = prob.cost([u_true, m_true, p])
print( "Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(total_cost, reg_cost, misfit_cost) )

# hl.nb.show_solution(
#     Vh, 
#     prob.u0, 
#     u, 
#     "Solution",
#     times=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
# )
# plt.show()

# hl.nb.show_solution(
#     Vh, 
#     prob.u0, 
#     u_true, 
#     "Solution",
#     times=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
# )
# plt.show()

# hl.nb.plot(prob.vec2func(m_true, hl.PARAMETER))
# plt.show()

# hl.nb.plot(prob.vec2func(m, hl.PARAMETER))
# plt.show()

import matplotlib.tri as tri

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

xy = mesh.coordinates()
triangulation = tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

vmin = min(m_true[:].min(), m[:].min())
vmax = max(m_true[:].max(), m[:].max())

vals_true = prob.vec2func(m_true, hl.PARAMETER).compute_vertex_values(mesh)
vals = prob.vec2func(m, hl.PARAMETER).compute_vertex_values(mesh)
axes[0].tripcolor(triangulation, vals_true, shading="gouraud", vmin=vmin, vmax=vmax, cmap="turbo")
axes[1].tripcolor(triangulation, vals, shading="gouraud", vmin=vmin, vmax=vmax, cmap="turbo")
plt.show()



H = hl.ReducedHessian(prob, misfit_only=True) 

k = 80
p = 20
print( "Single Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k, p))
Omega = hl.MultiVector(x[hl.PARAMETER], k+p)
hl.parRandom.normal(1., Omega)
lmbda, V = hl.singlePassG(H, prior.R, prior.Rsolver, Omega, k)

# posterior = hl.GaussianLRPosterior(prior, lmbda, V)

plt.plot(range(0,k), lmbda, 'b*', range(0,k+1), np.ones(k+1), '-r')
plt.yscale('log')
plt.xlabel('number')
plt.ylabel('eigenvalue')
plt.show()