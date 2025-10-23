from pathlib import Path

import torch
from torch import Tensor

from .models import SIRModel, SIRCompartmentModel


AUSTRIA_ADJACENCY_PATH = Path(__file__).resolve().parent.joinpath(
    "data", 
    "austria_adjacency.pt"
)


def load_austria_adjacency(device: torch.device = torch.device("cpu")) -> Tensor:
    """Reads in the data (failure distances (km) and censorship 
    information) used in @Dolgov2020.
    """
    austria_adjacency = torch.load(AUSTRIA_ADJACENCY_PATH).to(device=device)
    return austria_adjacency