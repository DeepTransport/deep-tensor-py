from collections import namedtuple
from pathlib import Path

import torch

from .preconditioner import GammaNormalMapping 


DATA_PATH = Path(__file__).resolve().parent.joinpath("data")
FAILURE_DISTS_PATH = DATA_PATH.joinpath("failure_dists.pt")
CENSORED_PATH = DATA_PATH.joinpath("censored.pt")


Data = namedtuple("Data", ["failure_dists", "censored"])


def load_shock_data(device: torch.device = torch.device("cpu")) -> Data:
    """Reads in the data (failure distances (km) and censorship 
    information) used in @Dolgov2020.
    """
    failure_dists = torch.load(FAILURE_DISTS_PATH).to(device=device)
    censored = torch.load(CENSORED_PATH).to(device=device)
    return Data(failure_dists, censored)