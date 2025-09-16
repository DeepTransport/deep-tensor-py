from dataclasses import dataclass

from ..verification import verify_method


TT_METHODS = ["random", "amen", "fixed_rank"]
INT_METHODS = ["deim", "maxvol"]


@dataclass
class TTOptions():
    r"""Options for configuring the construction of a TT object.
    
    Parameters
    ----------
    tt_method:
        The enrichment method used when constructing the TT cores. Can 
        be `'fixed_rank'` (no enrichment), `'random'`, or `'amen'` 
        [@Dolgov2014].
    int_method:
        The interpolation method used when constructing the tensor 
        cores. Can be `'maxvol'` [@Goreinov2010] or `'deim'` 
        [@Chaturantabut2010].
    max_als:
        The maximum number of ALS iterations to be carried out during 
        the FTT construction.
    init_rank:
        The initial rank of each tensor core.
    kick_rank:
        The rank of the enrichment set of samples added at each ALS 
        iteration.
    max_rank:
        The maximum allowable rank of each tensor core (prior to the 
        enrichment set being added).
    tol_svd:
        The threshold to use when applying truncated SVD to the tensor 
        cores when building the TT. The minimum number of singular 
        values such that the sum of their squares exceeds ($1-$ `tol_svd`) 
        will be retained.
    tol_max_core_error: 
        A stopping tolerance, $\epsilon$, based on the tensor cores. 
        More concretely, if $\boldsymbol{H}^{(\ell)}_{k}$ denotes the 
        $k$th tensor core during sweep $\ell$, the iterations are 
        stopped if
        $$
            \max_{k \in \{1, \dots, d\}}
                \frac{|\boldsymbol{H}^{(\ell)}_{k}-\boldsymbol{H}^{(\ell-1)}_{k}|_{\infty}}
                    {|\boldsymbol{H}^{(\ell)}_{k}|_{\infty}} < \epsilon.
        $$
    tol_l2_error:
        A stopping tolerance based on the estimated $L^{2}$ error of 
        the target function.
    verbose:
        If `verbose=0`, no information about the construction of the 
        FTT will be printed. If `verbose=1`, diagnostic information 
        will be prined at the end of each ALS iteration. If `verbose=2`, 
        the tensor core currently being constructed during each ALS 
        iteration will also be displayed.
    
    """
    
    tt_method: str = "amen"
    int_method: str = "maxvol"
    max_als: int = 1
    init_rank: int = 20
    kick_rank: int = 2
    max_rank: int = 30
    tol_svd: float = 0.0
    tol_max_core_error: float = 0.0
    tol_l2_error: float = 0.0
    verbose: int = 1
    
    def __post_init__(self):
        if self.kick_rank == 0:
            self.tt_method = "fixed_rank"
        self.tt_method = self.tt_method.lower()
        self.int_method = self.int_method.lower()
        verify_method(self.tt_method, TT_METHODS)
        verify_method(self.int_method, INT_METHODS)
        return