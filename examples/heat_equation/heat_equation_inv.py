import dolfin as dl
import hippylib as hl
from matplotlib import pyplot as plt
import numpy as np

from examples.heat_equation.heat_solver_old import HeatEquation, SpaceTimePointwiseStateObservation

mesh = dl.UnitSquareMesh(64, 64)

Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

# Generate true initial condition
ic_expr = dl.Expression(
    'std::min(0.5,std::exp(-100*(std::pow(x[0]-0.35,2) +  std::pow(x[1]-0.7,2))))',
    element=Vh.ufl_element())
true_initial_condition = dl.interpolate(ic_expr, Vh).vector()

# Generate prior
gamma = 1.0
delta = 8.0

# TODO: figure out if I can combine this into a single call to BiLaplacianPrior
prior = hl.BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)
prior.mean = dl.interpolate(dl.Constant(0.25), Vh).vector()

t_init = 0.0
t_final = 4.0
t_1 = 1.0
dt = 0.1
observation_dt = 0.2
    
simulation_times = np.arange(t_init, t_final + 0.5*dt, dt)
observation_times = np.arange(t_1, t_final + 0.5*dt, observation_dt)

targets = np.array([[0.0, 0.5], [0.1, 0.4]])

misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)

problem = HeatEquation(
    mesh=mesh,
    Vh=[Vh, Vh, Vh],
    prior=prior,
    misfit=misfit,
    simulation_times=simulation_times
)

# objs = [dl.Function(Vh,true_initial_condition),
#         dl.Function(Vh,prior.mean)]
# mytitles = ["True Initial Condition", "Prior mean"]
# hl.nb.multi1_plot(objs, mytitles)
# plt.show()

rel_noise = 0.01
utrue = problem.generate_vector(hl.STATE)
x = [utrue, true_initial_condition, None]
problem.solveFwd(x[hl.STATE], x)
misfit.observe(x, misfit.d)
MAX = misfit.d.norm("linf", "linf")
noise_std_dev = rel_noise * MAX
hl.parRandom.normal_perturb(noise_std_dev,misfit.d)
misfit.noise_variance = noise_std_dev*noise_std_dev

hl.nb.show_solution(Vh, true_initial_condition, utrue, "Solution")
plt.show()

m0 = true_initial_condition.copy()
_ = hl.modelVerify(problem, m0, is_quadratic=True)