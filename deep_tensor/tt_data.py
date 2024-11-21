import torch

from .directions import Direction, REVERSE_DIRECTIONS


class TTData():
    """Data associated with a functional TT approximation.
    
    Properties
    ----------
    direction:
        The direction in which the cores are being iterated over.
    cores:
        The tensor cores.
    interp_x: 
        TODO: write.
    res_x: 
        TODO (used with AMEN?)
    res_w:
        TODO: (used with AMEN?)
    rank:
        The ranks of each tensor core.

    """

    def __init__(self):

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
        ranks = torch.tensor([self.cores[k].shape[2] for k in range(num_dims)])
        
        return ranks

    def reverse_direction(self) -> None:
        """Reverses the direction in which the cores are iterated over.
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