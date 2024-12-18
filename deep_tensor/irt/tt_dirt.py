from typing import Callable

import torch

from .dirt import DIRT
from .tt_sirt import TTSIRT
from ..approx_bases import ApproxBases
from ..options import ApproxOptions


class TTDIRT(DIRT):

    def __init__(self, func: Callable, bases: ApproxBases, **kwargs):
        super().__init__(func, bases, **kwargs)

    def get_new_layer(
        self, 
        func: Callable, 
        bases: list[ApproxBases], 
        sirt_options: ApproxOptions, 
        xs: torch.Tensor, 
        neglogratio: torch.Tensor
    ) -> TTSIRT:
        """Func is the one that returns neglogpdf/negloglik."""

        def updated_func(zs: torch.Tensor) -> torch.Tensor:

            return self.bridge.ratio_func(
                func, 
                zs, 
                self.eval_irt, 
                self.reference, 
                self.dirt_options.method
            )

        if self.prev_approx is None:
            
            if self.num_layers <= 1:  # start from fresh (TODO: figure out why this happens on the second iteration?)
                
                # Generate debugging and initialisation samples
                input_data = self.get_inputdata(
                    bases[self.num_layers], 
                    xs, 
                    neglogratio
                )

                irt = TTSIRT(
                    updated_func, 
                    bases=bases[self.num_layers], 
                    options=sirt_options, 
                    input_data=input_data,
                    tau=self.dirt_options.defensive
                )

            else:

                # Start from the previous approximation
                input_data = self.get_inputdata(
                    self.irts[self.num_layers-1].approx.bases,
                    xs, 
                    neglogratio
                )

                irt = TTSIRT(
                    updated_func, 
                    bases=self.irts[self.num_layers-1].approx.bases,
                    approx=self.irts[self.num_layers-1].approx, 
                    options=sirt_options,
                    input_data=input_data,
                    tt_data=self.irts[self.num_layers-1].approx.data,
                    tau=self.dirt_options.defensive
                )
        
        else:
            raise NotImplementedError()
        
        return irt