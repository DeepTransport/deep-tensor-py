"""
Some things to do:
    - Define proper problem setup.
    - Figure out how to plot the parameters.
    - Build ROM?
"""

import multiprocessing
import time

import dolfin as dl
import hippylib as hl
from matplotlib import pyplot as plt
import numpy as np
import torch 
from torch import Tensor

import deep_tensor as dt

from solver import *
from plotting import *


torch.manual_seed(0)
np.random.seed(0)


mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(3.0, 1.0), 192, 64)
mesh_coarse = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(3.0, 1.0), 96, 32)

Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
Vh_coarse = dl.FunctionSpace(mesh_coarse, "Lagrange", 1)

## Define prior

# Define locations of Gaussian white noise
s0s = 3 * torch.linspace(1/18, 17/18, 9)
s1s = torch.linspace(1/6, 5/6, 3)
ss = torch.tensor([[s0, s1] for s0 in s0s for s1 in s1s])

mu = -5.0
r = 8.0
prior = ProcessConvolutionPrior(Vh, ss, mu=mu, r=r)
prior_coarse = ProcessConvolutionPrior(Vh_coarse, ss, mu=mu, r=r)

dt_reg = 0.25
dt_obs = 1.0
t_init = dt_reg
t_final = 10.0
t_1 = dt_obs
    
ts = np.arange(t_init, t_final+0.5*dt_reg, dt_reg)
ts_obs = np.arange(t_1, t_final+0.5*dt_reg, dt_obs)

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
misfit_coarse = SpaceTimePointwiseStateObservation(Vh_coarse, ts_obs, targets, var_error)

prob = HeatSolver(
    mesh=mesh,
    Vh=3 * [Vh],
    prior=prior,
    misfit=misfit,
    ts=ts
)

prob_coarse = HeatSolver(
    mesh=mesh_coarse,
    Vh=3 * [Vh_coarse],
    prior=prior_coarse,
    misfit=misfit_coarse,
    ts=ts
)

## Generate true parameters and observations

read_truth = True
read_data = True

if read_truth:
    m_true = np.load("m_true.npy")
else:
    m_true = torch.randn(prob.prior.dim)
    np.save("m_true.npy", m_true)

x = prob.generate_vector()
x[hl.PARAMETER] = m_true

prob.solveFwd(x[hl.STATE], x)

# x_crse = prob_coarse.generate_vector()
# x_crse[hl.PARAMETER] = m_true
# prob_coarse.solveFwd(x_crse[hl.STATE], x_crse)

# print(prob_fine.misfit.get_data(x))
# print(prob_crse.misfit.get_data(x_crse))

# print(prob_fine.misfit.get_data(x) - prob_crse.misfit.get_data(x_crse))

if read_data:
    d_obs = np.load("d_obs.npy")
else:
    d_obs = prob.misfit.get_data(x)
    np.save("d_obs.npy", d_obs)


def neglogpri(xs: Tensor) -> Tensor:
    """Returns the negative log prior density evaluated a given set of 
    samples.
    """
    return 0.5 * xs.square().sum(dim=1)


def __negloglik(i: int, x_i: Tensor) -> None:
        x = prob.generate_vector()
        x[hl.PARAMETER] = x_i 
        t0 = time.time()
        prob.solveFwd(x[hl.STATE], x)
        t1 = time.time()
        print(t1-t0)
        y = prob.misfit.get_data(x)
        # neglogliks[i] = 0.5 * (y - d_obs).square().sum() / var_error
        return 0.5 * (y - d_obs).square().sum() / var_error

def _negloglik(prob: HeatSolver, xs: Tensor) -> Tensor:
    """Returns the negative log-likelihood evaluated at a given set of 
    samples.
    """

    n_xs, _ = xs.shape
    neglogliks = torch.zeros(n_xs)
    u = prob.generate_vector(hl.STATE)

    # n_cpus = multiprocessing.cpu_count()
    # t0 = time.time()
    # p = multiprocessing.Pool(n_cpus)
    # print(list(enumerate(xs)))
    # neglogliks = p.starmap(__negloglik, enumerate(xs))
    # t1 = time.time()
    # print(torch.tensor(neglogliks))
    # print(f"Number of CPUs: {n_cpus}")
    # print(f"Number of solves: {xs.shape[0]}")
    # print(f"Total time (s): {t1-t0}")
    # print(f"Time per solve (s): {(t1-t0)/xs.shape[0]}")

    # raise Exception("stop...")

    for i, x_i in enumerate(xs):
        
        # t0 = time.time()
        x = [u, x_i, None]
        prob.solveFwd(x[hl.STATE], x)
        y = prob.misfit.get_data(x)
        # t1 = time.time()
        # print(t1-t0)

        neglogliks[i] = 0.5 * (y - d_obs).square().sum() / var_error
    
    return neglogliks


def negloglik(xs: Tensor) -> Tensor:
    return _negloglik(prob, xs)


def negloglik_coarse(xs: Tensor) -> Tensor:
    return _negloglik(prob_coarse, xs)


preconditioner = dt.IdentityPreconditioner(prior.dim)