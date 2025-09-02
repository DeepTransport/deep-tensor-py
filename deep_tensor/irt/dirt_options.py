from dataclasses import dataclass
from typing import Sequence


RATIO_METHODS = ["eratio", "aratio"]


def verify_method(method: str, accepted_methods: Sequence[str]) -> None:
        
    if method in accepted_methods:
        return 
    
    msg = (
        f"Method '{method}' not recognised. Expected one of: " 
        ", ".join(accepted_methods) + "."
    )
    raise ValueError(msg)


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
        The number of samples used to estimate the Hellinger divergence 
        between each bridging density and its DIRT approximation (and 
        to choose the parameters of each bridging density, if these are 
        being chosen adaptively).
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
    defensive: float = 1e-08
    cdf_tol: float = 1e-10
    verbose: float = 1
    
    def __post_init__(self):
        self.ratio_type = self.ratio_type.lower()
        verify_method(self.ratio_type, RATIO_METHODS)
        return