from dataclasses import dataclass

from .verification import verify_method


TT_METHODS = ["random", "amen", "fixed_rank"]
INT_METHODS = ["deim", "maxvol"]


@dataclass
class TTOptions():
    """Options for configuring the construction of an FTT object.
    
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
    local_tol:
        The threshold to use when applying truncated SVD to the tensor 
        cores when building the FTT.
    cdf_tol:
        The tolerance used when solving the root-finding problem to 
        invert the CDF. 
    tt_method:
        The method used to construct the TT cores. Can be `'fixed'`, 
        `'random'`, or `'amen'`.
    int_method:
        The interpolation method used when constructing the tensor 
        cores. Can be `'maxvol'` (Goreinov *et al.*, 2010) or `'deim'` 
        (Chaturantabut and Sorensen, 2010).
    verbose:
        If `verbose=0`, no information about the construction of the 
        FTT will be printed to the screen. If `verbose=1`, diagnostic 
        information will be prined at the end of each ALS iteration.
        If `verbose=2`, the tensor core currently being constructed 
        during each ALS iteration will also be displayed.

    References
    ----------
    Chaturantabut, S and Sorensen, DC (2010). *[Nonlinear model reduction 
    via discrete empirical interpolation](https://doi.org/10.1137/090766498)*. 
    SIAM Journal on Scientific Computing **32**, 2737--2764.

    Goreinov, SA, Oseledets, IV, Savostyanov, DV, Tyrtyshnikov, EE and 
    Zamarashkin, NL (2010). *[How to find a good submatrix](https://doi.org/10.1142/9789812836021_0015)*.
    In: Matrix Methods: Theory, Algorithms and Applications, 247--256.
    
    """
        
    max_als: int = 1
    als_tol: float = 1e-04
    init_rank: int = 20
    kick_rank: int = 2
    max_rank: int = 30
    local_tol: float = 1e-10
    cdf_tol: float = 1e-10
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