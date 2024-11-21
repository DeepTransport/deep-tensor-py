from .approx_options import ApproxOptions


TT_METHODS = ["random", "amen", "fixed_rank"]
INT_METHODS = ["qdeim", "wdeim", "maxvol"]


class TTOptions(ApproxOptions):

    def __init__(
        self, 
        max_als: int=4,
        als_tol: float=1e-4,
        init_rank: int=20,
        kick_rank: int=2,
        max_rank: int=30, 
        local_tol: float=1e-4,
        cdf_tol: float=1e-4,
        tt_method: str="amen",
        int_method: str="maxvol",
        verbose: bool=True
    ):

        # TODO: verify methods
        
        self.max_als = max_als
        self.als_tol = als_tol
        self.init_rank = init_rank
        self.kick_rank = kick_rank
        self.max_rank = max_rank
        self.local_tol = local_tol
        self.cdf_tol = cdf_tol
        self.tt_method = tt_method.lower()
        self.int_method = int_method.lower()
        self.verbose = verbose

        if self.kick_rank == 0:
            self.tt_method = "fixed_rank"

        return