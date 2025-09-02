from dataclasses import dataclass
from typing import Sequence


TT_METHODS = ["random", "amen", "fixed_rank"]
INT_METHODS = ["deim", "maxvol"]


def verify_method(method: str, accepted_methods: Sequence[str]) -> None:
        
    if method in accepted_methods:
        return 
    
    msg = (
        f"Method '{method}' not recognised. Expected one of: " 
        ", ".join(accepted_methods) + "."
    )
    raise ValueError(msg)


@dataclass
class TTOptions():
    """Options for configuring the construction of a TT object.
    
    Parameters
    ----------
    max_als:
        The maximum number of ALS iterations to be carried out during 
        the FTT construction.
    als_tol:
        The tolerance to use to determine whether the ALS iterations 
        should be terminated.
    init_rank:
        The initial rank of each tensor core.
    kick_rank:
        The rank of the enrichment set of samples added at each ALS 
        iteration.
    max_rank:
        The maximum allowable rank of each tensor core (prior to the 
        enrichment set being added).
    svd_tol:
        The threshold to use when applying truncated SVD to the tensor 
        cores when building the FTT. The minimum number of singular 
        values such that the sum of their squares exceeds ($1-$ `svd_tol`) 
        will be retained.
    tt_method:
        The enrichment method used when constructing the TT cores. Can 
        be `'fixed_rank'` (no enrichment), `'random'`, or `'amen'` 
        [@Dolgov2014].
    int_method:
        The interpolation method used when constructing the tensor 
        cores. Can be `'maxvol'` [@Goreinov2010] or `'deim'` 
        [@Chaturantabut2010].
    verbose:
        If `verbose=0`, no information about the construction of the 
        FTT will be printed. If `verbose=1`, diagnostic information 
        will be prined at the end of each ALS iteration. If `verbose=2`, 
        the tensor core currently being constructed during each ALS 
        iteration will also be displayed.
    
    """
        
    max_als: int = 1
    als_tol: float = 1e-01
    init_rank: int = 20
    kick_rank: int = 2
    max_rank: int = 30
    svd_tol: float = 0.0
    tt_method: str = "amen"
    int_method: str = "maxvol"
    verbose: int = 1
    
    def __post_init__(self):
        if self.kick_rank == 0:
            self.tt_method = "fixed_rank"
        self.tt_method = self.tt_method.lower()
        self.int_method = self.int_method.lower()
        verify_method(self.tt_method, TT_METHODS)
        verify_method(self.int_method, INT_METHODS)
        return