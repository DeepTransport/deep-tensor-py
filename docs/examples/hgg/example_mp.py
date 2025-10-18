from datetime import timedelta
import math
import time

import torch
import torch.multiprocessing as mp
from torch import Tensor

from tumortwin.models import ReactionDiffusion3D
from tumortwin.solvers import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.types import (
    ChemotherapySpecification, 
    RadiotherapySpecification
)
from tumortwin.utils import daterange

from examples.hgg.model import negloglik

import deep_tensor as dt

from utils import get_device, read_patient_data


torch.manual_seed(0)
device = get_device()


patient_data, measured_cellularity_maps, target_timepoints, target_solution = read_patient_data(
    folder_name="HGG_demo_001", 
    info_fname="HGG_demo_001.json", 
    num_visits_calibration=5, 
    plot=False
)

# Initial cellularity of tumor 
# (just using the first observation, could be improved)
u0 = torch.from_numpy(measured_cellularity_maps[0].array)

# Model Parameters: k = proliferation rate, d = diffusivity, theta = carrying capacity
k = torch.tensor(0.05, device=device)
d = torch.tensor(0.025, device=device)
theta = torch.tensor(1.0, device=device)

rt = RadiotherapySpecification(
    alpha=0.025,
    alpha_beta_ratio=10,
    times=[r.time for r in patient_data.radiotherapy],
    doses=[r.dose for r in patient_data.radiotherapy],
)

ct = ChemotherapySpecification(
    sensitivity=0.5,
    decay_rate=9.2420,
    times=[c.time for c in patient_data.chemotherapy],
    doses=[c.dose for c in patient_data.chemotherapy],
)

# Initialise model and solver

model = ReactionDiffusion3D(
    k=k,
    d=d,
    theta=theta,
    patient_data=patient_data,
    initial_time=patient_data.visits[0].time,
    chemotherapy_specifications=[ct],
    radiotherapy_specification=rt,
    require_grad=False
)

solver_options = TorchDiffEqSolverOptions(
    step_size=timedelta(days=0.5), 
    use_adjoint=True,
    device=device,
    method="rk4",
)

solver = TorchDiffEqSolver(model, solver_options)

timepoints = daterange(
    patient_data.visits[0].time, 
    patient_data.visits[-1].time, 
    timedelta(days=0.5)
)


def update_model_and_predict(model_parameters, timepoints=target_timepoints):
    
    d, k, alpha, ct_sens = torch.nn.Parameter(model_parameters)
    solver.model.d = torch.nn.Parameter(d)
    solver.model.k = torch.nn.Parameter(k)
    solver.model.radiotherapy_specification.alpha = torch.nn.Parameter(alpha)  # type: ignore
    solver.model.chemotherapy_specifications[0].sensitivity = torch.nn.Parameter(ct_sens)  # type: ignore

    _, predicted_cellularity_maps = solver.solve(timepoints=timepoints, u_initial=u0)
    return predicted_cellularity_maps


if __name__ == "__main__":

    print("running the main part of the script...")
    
    def neglogpost_alt(xs: torch.Tensor) -> torch.Tensor:
        """For now, let's just assume that the parameters are uniform 
        within the bounds.
        """

        neglogliks = torch.zeros_like(xs[:, 0]).share_memory_()

        num_threads = max(math.floor(mp.cpu_count() / xs.shape[0]), 1)
        torch.set_num_threads(num_threads)
        print(mp.cpu_count())

        t0 = time.time()

        processes = []
        for i in range(xs.shape[0]):
            args = (
                xs[i], 
                i,
                neglogliks, 
                target_solution, 
                solver, 
                u0, 
                target_timepoints
            )
            p = mp.Process(
                target=negloglik, 
                args=args
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

        # mp.Queue()

        print(neglogliks)
        total_time = time.time() - t0
        time_per_sim = total_time / xs.shape[0]

        print(f"{total_time = }")
        print(f"{time_per_sim = }")

        return neglogliks
    
    def neglogpost(xs: Tensor) -> Tensor:

        num_xs = xs.shape[0]
        # TEMP
        xs[0] = torch.tensor([0.1000, 0.0500, 0.0500, 0.2000])

        neglogliks = torch.zeros_like(xs[:, 0]).share_memory_()

        num_processes = min(mp.cpu_count(), num_xs)
        num_threads = max(math.floor(mp.cpu_count() / xs.shape[0]), 1)

        torch.set_num_threads(num_threads)
        print(mp.cpu_count())

        t0 = time.time()

        args = [
            [xs[i], i, neglogliks, target_solution, solver, u0, target_timepoints]
            for i in range(num_xs)
        ]

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(negloglik, args)

        # could use mp.Queue()??

        print(results)
        total_time = time.time()-t0
        time_per_sim = total_time / xs.shape[0]

        print(f"{total_time = }")
        print(f"{time_per_sim = }")

        return neglogliks

    # Could make prior a uniform distribution with these bounds...
    # TODO: evaluate the neglogpost at the true parameters and use this to rescale appropriately.
    # DONE: turns out it's basically zero
    # true parameters: [0.1000, 0.0500, 0.0500, 0.2000]
    bounds = torch.tensor([[0.05, 0.20], [0.025, 0.200], [0.01, 0.10], [0.10, 0.50]])
    dxs = bounds[:, 1] - bounds[:, 0]

    num_samples = 4
    dim = 4
    xs = torch.rand((num_samples, dim)) * dxs + bounds[:, 0]

    neglogpost(xs)

    print("hey...")

    #params = param_bounds.mean(dim=1)

    # update_model_and_predict(params)