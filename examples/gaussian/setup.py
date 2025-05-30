"""Builds a DIRT approximation to an ND Gaussian with an arbitrary mean 
and covariance matrix.
"""

import os

import torch

import deep_tensor as dt

from plotting import *
from random_gaussian import RandomGaussian


torch.manual_seed(1)

FOLDER_NAME = f"examples{os.sep}gaussian"


g = RandomGaussian(dim=4)

bounds = torch.tensor([-4.0, 4.0])
domain = dt.BoundedDomain(bounds=bounds)
reference = dt.GaussianReference(domain=domain)

preconditioner = dt.Preconditioner(
    reference, 
    g.Q, 
    g.Q_inv, 
    g.neglogdet_Q,
    g.neglogdet_Q_inv, 
    g.dim
)

poly = dt.Fourier(order=20)
dirt_options = dt.DIRTOptions()
tt_options = dt.TTOptions(tt_method="amen", max_rank=20, max_cross=2)
# bridge = dt.SingleLayer()
bridge = dt.Tempering()

dirt = dt.DIRT(
    g.negloglik, 
    g.neglogpri,
    preconditioner,
    poly, 
    bridge=bridge,
    tt_options=tt_options,
    dirt_options=dirt_options
)

save_path = f"{FOLDER_NAME}{os.sep}dirt"
dirt.save(save_path)

dirt = dt.SavedDIRT(save_path, preconditioner)

# saved_samples = saved_dirt.random(100)
# samples = dirt.random(100)



# import cProfile
# cProfile.run("""
# dirt = dt.TTDIRT(
#     g.negloglik, 
#     prior,
#     poly, 
#     bridge=bridge,
#     sirt_options=tt_options,
#     dirt_options=dirt_options
# )
# """, filename="test")

# import pstats
# from pstats import SortKey
# p = pstats.Stats('test')
# p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(100)