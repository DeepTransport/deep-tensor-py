# type: ignore

from datetime import timedelta
import pathlib
import time

from matplotlib import pyplot as plt
from pydantic import FilePath
import pyvista as pv
import torch
from torch import Tensor

from tumortwin.models.reaction_diffusion_3d import ReactionDiffusion3D
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

from docs.examples.hgg import MaternField3D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}.")

DATA_FOLDER = pathlib.Path("docs/examples/hgg/input_files")
PATIENT_INFO_PATH = FilePath(f"{DATA_FOLDER}/HGG_demo_001/HGG_demo_001.json")
IMAGE_PATH = FilePath(f"{DATA_FOLDER}/HGG_demo_001")

GRF_FOLDER = pathlib.Path(__file__).parent.joinpath("data", "HGG_demo_001").resolve()


crop_settings = CropSettings(
    crop_to=CropTarget.ROI_ENHANCE, 
    padding=10, 
    visit_index=-1
)

patient_data = HGGPatientData.from_file(
    PATIENT_INFO_PATH, 
    image_dir=IMAGE_PATH, 
    crop_settings=crop_settings
)

# plot_patient_timeline(patient_data)
# plot_imaging_summary(patient_data)

measured_cellularity_maps = [
    ADC_to_cellularity(
        visit.adc_image, visit.roi_enhance_image, visit.roi_nonenhance_image
    )
    for visit in patient_data.visits
]

def generate_mesh(shape, spacing):
    x_coords = torch.arange(0, shape[0]*spacing.x, spacing.x)
    y_coords = torch.arange(0, shape[1]*spacing.y, spacing.y)
    z_coords = torch.arange(0, shape[2]*spacing.z, spacing.z)
    x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    xx = x_grid.swapaxes(0, 2).numpy()
    yy = y_grid.swapaxes(0, 2).numpy()
    zz = z_grid.swapaxes(0, 2).numpy()
    mesh = pv.StructuredGrid(xx, yy, zz)
    return mesh 

img_shape = patient_data.brainmask_image.shape 
img_spacing = patient_data.brainmask_image.spacing
mesh = generate_mesh(img_shape, img_spacing)
field_logk = MaternField3D(mesh, ls=20, folder=GRF_FOLDER)

W = torch.randn(field_logk.num_points)
mean = -3.5 # TODO: modify?
sigma = 0.2
log_k = mean + field_logk.generate_field(W, sigma)

# mesh["log_proliferation_rate"] = x
# mesh["brain"] = patient_data.brainmask_image.array.flatten()
# mesh.set_active_scalars("brain")

# mesh_brain = mesh.extract_values(1.0)
# pv.plot(mesh_brain, scalars="log_proliferation_rate")

# Model Parameters: k = proliferation rate, d = diffusivity, theta = carrying capacity
k = torch.from_numpy(log_k).to(device).reshape(*img_shape).exp()
print(k.min())
print(k.max())
# k = torch.tensor(0.05, requires_grad=True, device=device)
d = torch.tensor(0.1, requires_grad=True, device=device) # 0.025
theta = torch.tensor(1.0, requires_grad=False, device=device)

rt = RadiotherapySpecification(
    alpha=0.05,
    alpha_beta_ratio=10,
    times=[r.time for r in patient_data.radiotherapy],
    doses=[r.dose for r in patient_data.radiotherapy],
)

ct = ChemotherapySpecification(
    sensitivity=0.2,
    decay_rate=9.2420,
    times=[c.time for c in patient_data.chemotherapy],
    doses=[c.dose for c in patient_data.chemotherapy],
)

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

timepoints = daterange(
    patient_data.visits[0].time, patient_data.visits[-1].time, timedelta(days=0.5)
)
# Initial condition for solver
u0 = torch.from_numpy(measured_cellularity_maps[0].array)

model = ReactionDiffusion3D(
    k=k,
    d=d,
    theta=theta,
    patient_data=patient_data,
    initial_time=patient_data.visits[0].time,
    chemotherapy_specifications=[ct],
    radiotherapy_specification=rt,
)

# def plot_slice(xs: Tensor, coords: Tensor, mask: Tensor, slice_ind: int):

#     xs = xs.reshape(mask.shape)
#     coords = coords.reshape(*mask.shape, 3)

#     xs = xs[:, :, slice_ind].flatten()
#     coords = coords[:, :, slice_ind, :2].reshape(-1, 2)
#     mask = mask[:, :, slice_ind].flatten()

#     xs = xs[mask==1] 
#     coords = coords[mask==1]

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.tripcolor(coords[:, 0], coords[:, 1], xs)

#     plt.show()
#     return

# plot_slice(x, model.voxel_coords, patient_data.brainmask_image.array, 17)

# pv.plot(mesh_brain, scalars="log_proliferation_rate")

solver_options = TorchDiffEqSolverOptions(
    step_size=timedelta(days=0.5),
    use_adjoint=True,
    device=device,
    method="rk4",
)

solver = TorchDiffEqSolver(model, solver_options)

# d, k, alpha, ct_sens = 0.05, 0.1, 0.05, 0.2
# solver.model.d = torch.nn.Parameter(d)
# solver.model.k = torch.nn.Parameter(k)
# solver.model.radiotherapy_specification.alpha = torch.nn.Parameter(alpha) 
# solver.model.chemotherapy_specifications[0].sensitivity = torch.nn.Parameter(ct_sens)

_, predicted_cellularity_maps = solver.solve(timepoints=timepoints, u_initial=u0)

import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(5,2))
plot_predicted_TCC(predicted_cellularity_maps, timepoints, ax=ax)
plot_measured_TCC(
    [m.array for m in measured_cellularity_maps],
    [v.time for v in patient_data.visits],
    ax=ax,
)
ax.legend(["predicted", "measured"]);

# plot cellularity maps for predictions and measurements
fig, axs = plt.subplots(2, len(patient_data.visit_days[::2]), figsize=(5,2))
for i, t in enumerate(patient_data.visit_days[::2]):
    time_days = np.array([days_since_first(t, timepoints[0]) for t in timepoints])
    t_idx = np.where(time_days == t)[0][0]
    plot_cellularity_map(
        predicted_cellularity_maps[t_idx], patient_data, time=t, ax=axs[0,i]
    )
    plot_cellularity_map(
        torch.tensor(measured_cellularity_maps[2*i].array), patient_data, time=t, ax=axs[1,i]
    )
axs[0,0].set_ylabel('Predicted')
axs[1,0].set_ylabel('Measured')
plt.show()


# def update_model_and_predict(model_parameters, timepoints=target_timepoints):
    
#     d, k, alpha, ct_sens = torch.nn.Parameter(model_parameters)
#     solver.model.d = torch.nn.Parameter(d)
#     solver.model.k = torch.nn.Parameter(k)
#     solver.model.radiotherapy_specification.alpha = torch.nn.Parameter(alpha) 
#     solver.model.chemotherapy_specifications[0].sensitivity = torch.nn.Parameter(ct_sens)

#     _, predicted_cellularity_maps = solver.solve(timepoints=timepoints, u_initial=u0)
#     return predicted_cellularity_maps

# sd_noise = 0.2

# def negloglik(params: Tensor) -> Tensor:
#     """Evaluates the negative log-likelihood function at each """

#     neglogliks = torch.zeros((params.shape[0],))

#     for i, params_i in enumerate(params):

#         t0 = time.time()

#         d, k, alpha, ct_sens = params_i

#         solver.model.d = torch.nn.Parameter(d)
#         solver.model.k = torch.nn.Parameter(k)
#         solver.model.radiotherapy_specification.alpha = torch.nn.Parameter(alpha)
#         solver.model.chemotherapy_specifications[0].sensitivity = torch.nn.Parameter(ct_sens)

#         predicted_cellularity_maps = solver.solve(target_timepoints, u0)[1]
#         nll = (1.0 / (2.0 * sd_noise ** 2)) * (predicted_cellularity_maps - target_solution).square().sum()
#         neglogliks[i] = nll

#         t1 = time.time()
#         print(f"Finished in {t1-t0:.4f} s.")
    
#     return neglogliks

# bounds = torch.tensor([[0.0, 2.0], [0.0, 0.5], [0.001, 0.1], [0.0, 1.0]])
# dxs = bounds[:, 1] - bounds[:, 0]

# num_samples = 16
# dim = 4
# xs = torch.rand((num_samples, dim)) * dxs + bounds[:, 0]

# negloglik(xs)