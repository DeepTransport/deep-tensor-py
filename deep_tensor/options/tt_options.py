from .approx_options import ApproxOptions


TT_METHODS = ["random", "amen", "fixed_rank"]
INT_METHODS = ["qdeim", "deim", "maxvol"]


class TTOptions(ApproxOptions):

    def __init__(
        self, 
        max_als: int=4,
        als_tol: float=1e-4,
        init_rank: int=20,
        kick_rank: int=2,
        max_rank: int=30, 
        local_tol: float=1e-10,
        cdf_tol: float=1e-10,
        tt_method: str="amen",
        int_method: str="maxvol",
        verbose: bool=True
    ):

        tt_method = tt_method.lower()
        int_method = int_method.lower()

        self._verify_method(tt_method, TT_METHODS)
        self._verify_method(int_method, INT_METHODS)
        
        self.max_als = max_als
        self.als_tol = als_tol
        self.init_rank = init_rank
        self.kick_rank = kick_rank
        self.max_rank = max_rank
        self.local_tol = local_tol
        self.cdf_tol = cdf_tol
        self.tt_method = "fixed_rank" if kick_rank == 0 else tt_method
        self.int_method = int_method
        self.verbose = verbose
        
        return