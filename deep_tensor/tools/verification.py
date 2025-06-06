from torch import Tensor 

import warnings


def check_finite(xs: Tensor) -> None:
    """Checks whether there are any NaN or inf values in a tensor, and 
    warns the user if so.
    """
    if (n_nans := xs.isnan().sum()) > 0:
        msg = f"{n_nans} NaN values detected."
        warnings.warn(msg)
    if (n_infs := xs.isinf().sum()) > 0:
        msg = f"{n_infs} inf values detected."
        warnings.warn(msg)
    return