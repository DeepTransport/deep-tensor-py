# type: ignore

import torch


def negloglik(args):

    params, i, neglogliks, target_solution, solver, u0, timepoints = args

    print(f"Simulation {i}")
    
    d, k, alpha, ct_sens = torch.nn.Parameter(params)
    solver.model.d = torch.nn.Parameter(d)
    solver.model.k = torch.nn.Parameter(k)
    solver.model.radiotherapy_specification.alpha = torch.nn.Parameter(alpha)
    solver.model.chemotherapy_specifications[0].sensitivity = torch.nn.Parameter(ct_sens)

    _, predicted_cellularity_maps = solver.solve(timepoints, u0)
    nll = 0.5 * (predicted_cellularity_maps - target_solution).square().sum()
    neglogliks[i] = nll.detach()
    return neglogliks[i]