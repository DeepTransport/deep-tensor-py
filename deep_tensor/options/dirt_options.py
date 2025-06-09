from dataclasses import dataclass

from .verification import verify_method


METHODS = ["eratio", "aratio"]


@dataclass
class DIRTOptions():
    r"""Options for configuring the construction of a DIRT object.

    Parameters
    ----------
    method:
        The method used for the ratio function at each iteration. Can 
        be `'aratio'` (approximate ratio) or `'eratio'` (exact ratio).
    num_samples:
        The number of samples generated to be used as part of the 
        construction of the DIRT.
    num_debugs:
        The number of samples used to evaluate the quality of each SIRT 
        constructed during the construction of the DIRT.
    defensive:
        The parameter (often referred to as $\gamma$ or $\tau$) used to 
        make the tails of the FTT approximation to each ratio function 
        heavier.
    verbose:
        Whether to print information on the construction of the DIRT 
        object.
    
    """

    method: str = "aratio"
    num_samples: int = 1000
    num_debugs: int = 1000
    defensive: float = 1e-08
    verbose: bool = True
        
    def __post_init__(self):
        self.method = self.method.lower()
        verify_method(self.method, METHODS)
        return