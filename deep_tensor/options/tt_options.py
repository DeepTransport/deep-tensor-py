from dataclasses import dataclass

from .verification import verify_method


TT_METHODS = ["random", "amen", "fixed_rank"]
INT_METHODS = ["qdeim", "deim", "maxvol"]


@dataclass
class TTOptions():
    """Options for configuring the construction of an FTT object.
    
    Parameters
    ----------
    max_cross:
        The maximum number of cross iterations to be carried out during 
        the FTT construction.
    cross_tol:
        TODO
    init_rank:
        The initial rank of each tensor core.
    kick_rank:
        The rank of the enrichment set of samples added at each cross 
        iteration.
    max_rank:
        The maximum allowable rank of each tensor core.
    local_tol:
        TODO
    cdf_tol:
        The tolerance used when solving the root-finding problem to 
        invert the CDF. 
    tt_method:
        The method used to construct the TT cores. Can be `fixed`, 
        `random`, or `amen`.
    int_method:
        The interpolation method used when constructing the tensor 
        cores. Can be `maxvol`, `qdeim`, or `deim`.
    verbose:
        Whether to print the results of FTT construction to the screen 
        at each iteration.
    
    """
        
    max_cross: int = 4
    cross_tol: float = 1e-04
    init_rank: int = 20
    kick_rank: int = 2
    max_rank: int = 30
    local_tol: float = 1e-10
    cdf_tol: float = 1e-10
    tt_method: str = "amen"
    int_method: str = "maxvol"
    verbose: bool = True
    
    def __post_init__(self):
        if self.kick_rank == 0:
            self.tt_method = "fixed_rank"
        self.tt_method = self.tt_method.lower()
        self.int_method = self.int_method.lower()
        verify_method(self.tt_method, TT_METHODS)
        verify_method(self.int_method, INT_METHODS)
        return