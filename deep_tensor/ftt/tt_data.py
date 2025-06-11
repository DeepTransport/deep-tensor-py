import torch
from torch import Tensor

from .directions import Direction, REVERSE_DIRECTIONS


class TTData():
    """Data associated with a functional tensor train."""

    def __init__(
        self, 
        direction: Direction = Direction.FORWARD,
        cores: dict[int, Tensor] | None = None
    ):
        if cores is None:
            cores = {}
        self.direction = direction
        self.cores: dict[int, Tensor] = cores
        self.interp_ls: dict[int, Tensor] = {}
        self.res_x: dict[int, Tensor] = {}  # Residual coordinates for AMEN
        self.res_w: dict[int, Tensor] = {}  # Residual blocks for AMEN
        return
    
    @property
    def rank(self) -> Tensor:
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