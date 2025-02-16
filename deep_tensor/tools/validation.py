from torch import Tensor 

import warnings


def check_for_nans(xs: Tensor) -> None:
    """Checks whether there are any NAN or INF values in a tensor, and 
    warns the user if so.
    """
    if (n_nans := xs.isnan().sum()) > 0:
        msg = f"{n_nans} NAN values detected."
        warnings.warn(msg)
    if (n_nans := xs.isinf().sum()) > 0:
        msg = f"{n_nans} INF values detected."
        warnings.warn(msg)
    return