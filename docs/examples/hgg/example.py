import os
from datetime import timedelta
import math
import pathlib
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from pydantic import FilePath
from rich import print

from tumortwin.models import ReactionDiffusion3D
from tumortwin.optimizers import LMoptimizer, LMoptions
from tumortwin.postprocessing import (
    compute_total_cell_count,
    plot_calibration,
    plot_calibration_iter,
    plot_cellularity_map,
    plot_imaging_summary,
    plot_loss,
    plot_measured_TCC,
    plot_patient_timeline,
    plot_predicted_TCC,
)
from tumortwin.preprocessing import ADC_to_cellularity, compute_carrying_capacity
from tumortwin.solvers import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.types import (
    ChemotherapySpecification,
    CropSettings,
    CropTarget,
    RadiotherapySpecification,
)
from tumortwin.types.hgg_data import HGGPatientData
from tumortwin.utils import daterange, days_since_first

from examples.hgg.model import negloglik


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_FOLDER = pathlib.Path(__file__).resolve().parent.joinpath("input_files")
PATIENT_INFO_PATH = FilePath(f"{DATA_FOLDER}/HGG_demo_001/HGG_demo_001.json")  # type: ignore
IMAGE_PATH = FilePath(f"{DATA_FOLDER}/HGG_demo_001")  # type: ignore

crop_settings = CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10, visit_index=-1)
patient_data = HGGPatientData.from_file(
    PATIENT_INFO_PATH, image_dir=IMAGE_PATH, crop_settings=crop_settings
)

# plot_patient_timeline(patient_data)
# plot_imaging_summary(patient_data)

measured_cellularity_maps = [
    ADC_to_cellularity(
        visit.adc_image, visit.roi_enhance_image, visit.roi_nonenhance_image
    )
    for visit in patient_data.visits
]

# Model Parameters: k = proliferation rate, d = diffusivity, theta = carrying capacity
k = torch.tensor(0.05, requires_grad=True, device=device)
d = torch.tensor(0.025, requires_grad=True, device=device)
theta = torch.tensor(1.0, requires_grad=False, device=device)

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

model = ReactionDiffusion3D(
    k=k,
    d=d,
    theta=theta,
    patient_data=patient_data,
    initial_time=patient_data.visits[0].time,
    chemotherapy_specifications=[ct],
    radiotherapy_specification=rt,
)

solver_options = TorchDiffEqSolverOptions(
    step_size=timedelta(days=0.5),
    use_adjoint=True,
    device=device,
    method="rk4",
)

solver = TorchDiffEqSolver(model, solver_options)

timepoints = daterange(
    patient_data.visits[0].time, patient_data.visits[-1].time, timedelta(days=0.5)
)
u0 = torch.from_numpy(measured_cellularity_maps[0].array)

# import time 
# t0 = time.time()
# times, predicted_cellularity_maps = solver.solve(timepoints=timepoints, u_initial=u0)
# print(time.time()-t0)

# How many imaging dates do we want to try and match
n_visits_calibration = 5  # *Including* the initial visit

target_timepoints = [visit.time for visit in patient_data.visits[:n_visits_calibration]]
target_solution = torch.stack(
    tuple(
        [
            torch.from_numpy(m.array)
            for m in measured_cellularity_maps[: n_visits_calibration]
        ]
    )
)


def update_model_and_predict(model_parameters, timepoints = target_timepoints):
    
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
        total_time = time.time()-t0
        time_per_sim = total_time / xs.shape[0]

        print(f"{total_time = }")
        print(f"{time_per_sim = }")

        return neglogliks
    
    def neglogpost(xs: torch.Tensor) -> torch.Tensor:

        num_xs = xs.shape[0]

        neglogliks = torch.zeros_like(xs[:, 0]).share_memory_()

        num_processes = min(mp.cpu_count(), num_xs)
        num_threads = max(math.floor(mp.cpu_count() / xs.shape[0]), 1)

        torch.set_num_threads(num_threads)
        print(mp.cpu_count())

        t0 = time.time()

        args = [
            [xs[i], i,neglogliks, target_solution, solver, u0, target_timepoints]
            for i in range(num_xs)
        ]

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(negloglik, args)

        # mp.Queue()

        print(results)
        total_time = time.time()-t0
        time_per_sim = total_time / xs.shape[0]

        print(f"{total_time = }")
        print(f"{time_per_sim = }")

        return neglogliks

    # could make prior a uniform distribution with these bounds..
    bounds = torch.tensor([[0.0, 2.0], [0.0, 0.5], [0.001, 0.1], [0.0, 1.0]])
    dxs = bounds[:, 1] - bounds[:, 0]

    num_samples = 16
    dim = 4
    xs = torch.rand((num_samples, dim)) * dxs + bounds[:, 0]

    neglogpost(xs)

    print("hey...")

    #params = param_bounds.mean(dim=1)

    # update_model_and_predict(params)