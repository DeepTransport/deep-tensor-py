from dataclasses import dataclass

from ..verification import verify_method


RATIO_METHODS = ["eratio", "aratio"]


@dataclass
class DIRTOptions():
    r"""Options for configuring the construction of a DIRT object.
    
    Parameters
    ----------
    ratio_type: 
        Whether to approximate the approximate ratio function (`'aratio'`) 
        or the exact ratio function (`'eratio'`) when constructing each 
        layer of the DIRT. 
    num_error_samples:
        The number of samples used to estimate the Hellinger distance 
        between each bridging density and its DIRT approximation, (and 
        to choose the parameters of each bridging density, if these are 
        being chosen adaptively).
    num_error_samples_ratio:
        The number of samples used to estimate the Hellinger distance 
        between each ratio function and its SIRT approximation.
    num_error_samples_ratio_red:
        The number of samples used to estimate the Hellinger distance 
        between each reduced ratio function and its SIRT approximation. 
        This should only be nonzero if a subspace is being used to 
        construct each SIRT.
    defensive:
        The defensive term (often referred to as $\gamma$ or $\tau$) 
        used to make the tails of the DIRT approximation to the target 
        density heavier. 
    cdf_tol:
        The numerical tolerance used when evaluating the inverse CDFs 
        required to evaluate the (deep) inverse Rosenblatt transport.
    verbose:
        If `verbose=0`, no information about the construction of the 
        DIRT will be printed. If `verbose=1`, diagnostic information 
        will be displayed after the construction of each DIRT layer.
    
    """
        
    ratio_type: str = "aratio"
    num_error_samples: int = 1000
    num_error_samples_ratio: int = 0
    num_error_samples_ratio_red: int = 0
    defensive: float = 1e-08
    cdf_tol: float = 1e-12
    verbose: float = 1
    
    def __post_init__(self):
        self.ratio_type = self.ratio_type.lower()
        verify_method(self.ratio_type, RATIO_METHODS)
        return