import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
import ufl
# sys.path.append(os.environ.get('HIPPYLIB_BASE_DIR', "../"))
from hippylib import *

import logging

#logging.getLogger('FFC').setLevel(logging.WARNING)
#logging.getLogger('UFL').setLevel(logging.WARNING)
#dl.set_log_active(False)

np.random.seed(seed=1)

def true_model(pri: modeling._Prior):

    noise = dl.Vector()
    pri.init_vector(noise, "noise")
    parRandom.normal(1.0, noise)  # iid Gaussian noise

    mtrue = dl.Vector()
    pri.init_vector(mtrue, 0)
    pri.sample(noise, mtrue)  # sets mtrue to a sample from the prior

    return mtrue

ndim = 2
nx = 64
ny = 64
mesh = dl.UnitSquareMesh(nx, ny)
Vh2 = dl.FunctionSpace(mesh, "Lagrange", 2)
Vh1 = dl.FunctionSpace(mesh, "Lagrange", 1)
Vh = [Vh2, Vh1, Vh2]

print(f"Number of dofs: " + 
      f"STATE = {Vh[STATE].dim()}, " + 
      f"PARAMETER = {Vh[PARAMETER].dim()}, " + 
      f"ADJOINT = {Vh[ADJOINT].dim()}")

def u_boundary(x: np.ndarray, on_boundary: bool) -> bool:
    """Returns a boolean that indicates whether a point is located on 
    the top or bottom boundary of the domain.
    """
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

# Boundary conditions for forward and adjoint problems
u_bdr = dl.Expression("x[1]", degree=1)
u_bdr0 = dl.Constant(0.0)
bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)

f = dl.Constant(0.0)

def pde_varf(u, m, p):
    """Returns the variational form of the PDE.
    
    Parameters
    ----------
    u:
        State variable.
    m: 
        Parameter.
    p: 
        Adjoint variable.
    
    """
    return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx - f * p * ufl.dx

# bc = boundary conditions for the forward problem
# bc0 = boundary conditions for the adjoint and incremental problems
# This class solves the forward/adjoint/incremental problems, and 
# evaluates the first and second partial derivatives of the forward 
# problem with respect to the state/parameter/adjoint variables
pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

# Define covariance parameters
gamma = 0.1
delta = 0.5

# Define anisotropy parameters
# theta0 = 2.0
# theta1 = 0.5
# alpha = np.pi / 4.0

# anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree=1)
# anis_diff.set(theta0, theta1, alpha)

pri = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, robin_bc=True) # anis_diff
mtrue = true_model(pri)

print("Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma, 2))    

# Plot truth and prior mean
objs = [dl.Function(Vh[PARAMETER], mtrue), dl.Function(Vh[PARAMETER], pri.mean)]
mytitles = ["True Parameter", "Prior mean"]
nb.multi1_plot(objs, mytitles)
plt.show()

ntargets = 50
rel_noise = 0.01

# Define observation locations
# Targets only on the bottom
targets_x = np.random.uniform(0.1, 0.9, [ntargets])
targets_y = np.random.uniform(0.1, 0.5, [ntargets])
targets = np.zeros([ntargets, ndim])
targets[:, 0] = targets_x
targets[:, 1] = targets_y
# targets everywhere
# targets = np.random.uniform(0.1,0.9, [ntargets, ndim] )
print("Number of observation points: {0}".format(ntargets))
obs = PointwiseStateObservation(Vh[STATE], targets)

utrue = pde.generate_state()
x = [utrue, mtrue, None]
pde.solveFwd(x[STATE], x)
obs.B.mult(x[STATE], obs.d)
MAX = obs.d.norm("linf")
noise_std_dev = rel_noise * MAX
parRandom.normal_perturb(noise_std_dev, obs.d)
obs.noise_variance = noise_std_dev*noise_std_dev

vmax = max( utrue.max(), obs.d.max() )
vmin = min( utrue.min(), obs.d.min() )

plt.figure(figsize=(15,5))
nb.plot(dl.Function(Vh[STATE], utrue), mytitle="True State", subplot_loc=121, vmin=vmin, vmax=vmax)
nb.plot_pts(targets, obs.d, mytitle="Observations", subplot_loc=122, vmin=vmin, vmax=vmax)
plt.show()

model = Model(pde, pri, obs)

m0 = dl.interpolate(dl.Expression("sin(x[0])", degree=5), Vh[PARAMETER])
_ = modelVerify(model, m0.vector())

## Compute the MAP point

m = pri.mean.copy()
solver = ReducedSpaceNewtonCG(model)
solver.parameters["rel_tolerance"] = 1e-6
solver.parameters["abs_tolerance"] = 1e-12
solver.parameters["max_iter"]      = 25
solver.parameters["GN_iter"] = 5
solver.parameters["globalization"] = "LS"
solver.parameters["LS"]["c_armijo"] = 1e-4


x = solver.solve([None, m, None])

if solver.converged:
    print( "\nConverged in ", solver.it, " iterations.")
else:
    print( "\nNot Converged")

print( "Termination reason: ", solver.termination_reasons[solver.reason] )
print( "Final gradient norm: ", solver.final_grad_norm )
print( "Final cost: ", solver.final_cost )

plt.figure(figsize=(15,5))
nb.plot(dl.Function(Vh[STATE], x[STATE]), subplot_loc=121,mytitle="State")
nb.plot(dl.Function(Vh[PARAMETER], x[PARAMETER]), subplot_loc=122,mytitle="Parameter")
plt.show()

model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
Hmisfit = ReducedHessian(model, misfit_only=True)
k = 50
p = 20
print( "Single/Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )

Omega = MultiVector(x[PARAMETER], k+p)
parRandom.normal(1., Omega)
lmbda, V = doublePassG(Hmisfit, pri.R, pri.Rsolver, Omega, k)

post = GaussianLRPosterior(pri, lmbda, V)
post.mean = x[PARAMETER]

plt.plot(range(0,k), lmbda, 'b*', range(0,k+1), np.ones(k+1), '-r')
plt.yscale('log')
plt.xlabel('number')
plt.ylabel('eigenvalue')

# nb.plot_eigenvectors(Vh[PARAMETER], V, mytitle="Eigenvector", which=[0,1,2,5,10,15])

compute_trace = True
if compute_trace:
    post_tr, prior_tr, corr_tr = post.trace(method="Randomized", r=200)
    print( "Posterior trace {0:5e}; Prior trace {1:5e}; Correction trace {2:5e}".format(post_tr, prior_tr, corr_tr) )
post_pw_variance, pr_pw_variance, corr_pw_variance = post.pointwise_variance(method="Randomized", r=200)

objs = [dl.Function(Vh[PARAMETER], pr_pw_variance),
        dl.Function(Vh[PARAMETER], post_pw_variance)]
mytitles = ["Prior variance", "Posterior variance"]
nb.multi1_plot(objs, mytitles, logscale=False)
plt.show()

nsamples = 5
noise = dl.Vector()
post.init_vector(noise,"noise")
s_prior = dl.Function(Vh[PARAMETER], name="sample_prior")
s_post = dl.Function(Vh[PARAMETER], name="sample_post")

pr_max   =  2.5*np.sqrt( pr_pw_variance.max() ) + pri.mean.max()
pr_min   = -2.5*np.sqrt( pr_pw_variance.max() ) + pri.mean.min()
ps_max   =  2.5*np.sqrt( post_pw_variance.max() ) + post.mean.max()
ps_min   = -2.5*np.sqrt( post_pw_variance.max() ) + post.mean.min()

for i in range(nsamples):
    parRandom.normal(1., noise)
    post.sample(noise, s_prior.vector(), s_post.vector())
    plt.figure(figsize=(15,5))
    nb.plot(s_prior, subplot_loc=121,mytitle="Prior sample", vmin=pr_min, vmax=pr_max)
    nb.plot(s_post, subplot_loc=122,mytitle="Posterior sample", vmin=ps_min, vmax=ps_max)
    plt.show()