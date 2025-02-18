from dataclasses import dataclass

from .verification import verify_method


TT_METHODS = ["random", "amen", "fixed_rank"]
INT_METHODS = ["qdeim", "deim", "maxvol"]


@dataclass
class TTOptions():
        
    max_als: int = 4
    als_tol: float = 1e-4
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