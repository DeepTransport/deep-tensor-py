from typing import Dict

import torch
from torch import Tensor

from .directions import Direction, REVERSE_DIRECTIONS


class TTData():
    """Data associated with a functional tensor train."""

    def __init__(
        self,
        direction: Direction | None = None,
        coefs: Dict[int, Tensor] | None = None,
        cores: Dict[int, Tensor] | None = None
    ):

        if direction is None: 
            self.direction = Direction.FORWARD
        else:
            self.direction = direction

        # Define the coefficient tensors of the TT decomposition, and 
        # the coefficient tensors of the FTT cores
        if coefs is None:
            self.coefs: Dict[int, Tensor] = {}
        else:
            self.coefs = coefs
        
        if cores is None:
            self.cores: Dict[int, Tensor] = {}
        else:
            self.cores = cores

        # Define interpolation point sets in each dimension 
        self.interp_ls: Dict[int, Tensor] = {}

        # Define residual coordinates and blocks for AMEN
        self.res_x: Dict[int, Tensor] = {}
        self.res_w: Dict[int, Tensor] = {}

        return
    
    @property
    def _rank(self) -> Tensor:
        """The ranks of each tensor core."""
        ranks = [self.cores[k].shape[2] for k in range(len(self.cores))]
        return torch.tensor(ranks)

    def _reverse_direction(self) -> None:
        """Reverses the direction in which the dimensions of the 
        function are iterated over.
        """
        self.direction = REVERSE_DIRECTIONS[self.direction]
        return

    def _clean(self) -> None:
        """Removes all of the intermediate data used to build the 
        tensor train (but retains the cores and evaluation direction).
        """
        self.interp_ls = {}
        self.res_x = {}
        self.res_w = {}
        return