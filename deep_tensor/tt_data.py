import torch

from .directions import Direction, REVERSE_DIRECTIONS


class TTData():

    def __init__(self):
        """Data associated with a functional TT approximation."""

        self.direction = Direction.FORWARD 
        self.cores: dict[int, torch.Tensor] = {}
        self.interp_x: dict[int, torch.Tensor] = {}
        
        self.res_x = {}
        self.res_w = {}
        
        return
    
    @property
    def rank(self) -> torch.Tensor:
        """The ranks of each tensor core."""
        num_dims = len(self.cores)
        ranks = [self.cores[k].shape[2] for k in range(num_dims)]
        return torch.tensor(ranks)

    def reverse_direction(self) -> None:
        """Reverses the direction in which the dimensions of the 
        function are iterated over.
        """
        self.direction = REVERSE_DIRECTIONS[self.direction]
        return

    def clean(self) -> None:
        """Removes all of the intermediate data used to build the 
        tensor train (but retains the cores and evaluation direction).
        """
        self.interp_x = torch.tensor([])
        self.res_x = {}
        self.res_w = {}
        return