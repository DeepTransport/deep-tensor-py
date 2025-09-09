from pathlib import Path

import torch

from .models import SIRModel, SIRCompartmentModel


AUSTRIA_PATH = Path(__file__).resolve().parent.joinpath(
    "data", 
    "austria_adjacency.pt"
)
austria_adjacency = torch.load(AUSTRIA_PATH)