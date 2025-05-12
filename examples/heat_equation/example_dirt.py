"""
Some things to do:
    - Define proper problem setup.
    - Figure out how to plot the parameters.
    - Build ROM?
"""

import time

import dolfin as dl
import hippylib as hl
from matplotlib import pyplot as plt
import numpy as np
import torch 
from torch import Tensor

from solver import *

import deep_tensor as dt


mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(3.0, 1.0), 64, 64)

Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

## Define prior

# Define locations of Gaussian white noise
s0s = 3 * torch.linspace(1/18, 17/18, 9)
s1s = torch.linspace(1/6, 5/6, 3)
ss = torch.tensor([[s0, s1] for s0 in s0s for s1 in s1s])

prior = ProcessConvolutionPrior(
    Vh, 
    ss, 
    mu=-5.0, 
    r=8.0
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

std_error = 1.65e-2
var_error = std_error ** 2
misfit = SpaceTimePointwiseStateObservation(Vh, ts_obs, targets, var_error)

prob = HeatSolver(
    mesh=mesh,
    Vh=[Vh, Vh, Vh],
    prior=prior,
    misfit=misfit,
    ts=ts
)

## Generate true parameters and observations

m_true = torch.normal(mean=0.0, std=1.0, size=(prob.prior.n_coefs,))
u_true = prob.generate_vector(hl.STATE)
x = [u_true, m_true, None]

prob.solveFwd(x[hl.STATE], x)
d_obs = prob.misfit.get_data(x)


# print(", ".join([str(d)[:6] for d in d_obs.tolist()]))

# hl.nb.plot(prob.prior.transform(m_true), hl.PARAMETER) 
# plt.show()


def neglogpri(xs: Tensor) -> Tensor:
    """Returns the negative log prior density evaluated a given set of 
    samples.
    """
    return 0.5 * xs.sum().square(dim=1)


def negloglik(xs: Tensor) -> Tensor:
    """Returns the negative log-likelihood evaluated at a given set of 
    samples.
    """

    n_xs, _ = xs.shape
    neglogliks = torch.zeros(n_xs)
    u = prob.generate_vector(hl.STATE)
    
    for i, x_i in enumerate(xs):
        
        x = [u, x_i, None]
        prob.solveFwd(x[hl.STATE], x)
        y = prob.misfit.get_data(x)

        neglogliks[i] = 0.5 * (y - d_obs).square().sum() / var_error
    
    return neglogliks