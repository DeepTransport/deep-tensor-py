from typing import Callable

import torch

from .approx_bases import ApproxBases


def get_potential_to_density(
    base: ApproxBases, 
    y, 
    z
):
    
    _, dxdz = base.reference2domain(base, z)
    
    # log density of the reference measure
    mlogw = base.eval_measure_potential_reference(base, z)
    y = torch.exp(-0.5 * (y - mlogw - torch.sum(torch.log(dxdz), 0)))

    return y
        
def potential_to_density(
    base: ApproxBases, 
    potential_fun: Callable, 
    z
):

    x, dxdz = base.reference2domain(base, z)
    y = feval(potential_fun, x)
    
    # log density of the reference measure
    mlogw = base.eval_measure_potential_reference(base, z)
    y = torch.exp(-0.5*(y - mlogw - torch.sum(torch.log(dxdz), 0)))
    return y