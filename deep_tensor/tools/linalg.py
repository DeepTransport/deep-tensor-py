from typing import Tuple

import torch


def reshape_matlab(A: torch.Tensor, newshape: Tuple) -> torch.Tensor:
    """https://stackoverflow.com/questions/63960352/reshaping-order-in-pytorch-fortran-like-index-ordering"""
    
    A = (A.permute(*reversed(range(A.ndim)))
          .reshape(*reversed(newshape))
          .permute(*reversed(range(len(newshape)))))
    
    return A