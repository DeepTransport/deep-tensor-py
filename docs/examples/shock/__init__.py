import torch

from .preconditioner import GammaNormalMapping


# Define failure distances (km)
failure_dists = torch.tensor([
    6700,  6950,  7820,  8790,  9120,
    9660,  9820,  11310, 11690, 11850, 
    11880, 12140, 12200, 12870, 13150, 
    13330, 13470, 14040, 14300, 17520,
    17540, 17890, 18420, 18960, 18980,
    19410, 20100, 20100, 20150, 20320, 
    20900, 22700, 23490, 26510, 27410, 
    27490, 27890, 28100
])

# Define whether or not each observation is right-censored
censored = torch.tensor([
    False, True,  True,  True,  False, 
    True,  True,  True,  True,  True, 
    True,  True,  False, True,  False, 
    True,  True,  True,  False, False, 
    True,  True,  True,  True,  True, 
    True,  False, True,  True,  True, 
    False, False, True,  False, True, 
    False, True,  True
])