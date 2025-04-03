import dolfin as dl
import hippylib as hl
import numpy as np

from model_ad_diff import TimeDependentAD, SpaceTimePointwiseStateObservation

mesh = dl.UnitSquareMesh(64, 64)

Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

# Generate prior
gamma = 1.0
delta = 8.0
prior = hl.BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)

t_init = 0.0
t_final = 4.0
t_1 = 1.0
dt = 0.1
observation_dt = 0.2
    
simulation_times = np.arange(t_init, t_final + 0.5*dt, dt)
observation_times = np.arange(t_1, t_final + 0.5*dt, observation_dt)

targets = np.array([[0.0, 0.5], [0.1, 0.4]])

misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)

ad = TimeDependentAD(
    mesh=mesh,
    Vh=Vh,
    prior=prior,
    misfit=misfit,
    simulation_times=simulation_times,
    wind_velocity=None,
    gls_stab=True
)