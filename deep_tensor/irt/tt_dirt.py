from typing import Callable

import torch

from .dirt import DIRT
from .tt_sirt import TTSIRT
from ..approx_bases import ApproxBases
from ..options import ApproxOptions

from copy import deepcopy


class TTDIRT(DIRT):

    def __init__(self, func: Callable, bases: ApproxBases, **kwargs):
        super().__init__(func, bases, **kwargs)

    def get_new_layer(
        self, 
        func: Callable, 
        bases: list[ApproxBases], 
        sirt_options: ApproxOptions, 
        xs: torch.Tensor, 
        neglogratios: torch.Tensor
    ) -> TTSIRT:
        """TODO: write docstring.

        Parameters
        ----------
        func:
            Function that returns the negative log-likelihood and 
            negative log-prior density of a sample.
        bases:
            List of approximation bases in each dimension.
        sirt_options:
            Options used when constructing the SIRT associated with the 
            layer.
        xs:
            An n * d matrix containing samples distributed according to
            the current bridging density.
        neglogratios:
            An n-dimensional vector containing the negative log-ratio 
            function evaluated at each element in xs.

        Returns
        -------
        sirt:
            The squared inverse Rosenblatt transport approximation to 
            the next bridging density.
        
        """

        def updated_func(rs: torch.Tensor) -> torch.Tensor:

            return self.bridge.ratio_func(
                func, 
                rs, 
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
                    neglogratios
                )

                sirt = TTSIRT(
                    updated_func, 
                    bases=bases[self.num_layers], 
                    options=sirt_options, 
                    input_data=input_data,
                    tau=self.dirt_options.defensive
                )

            else:

                # Start from the previous approximation
                input_data = self.get_inputdata(
                    deepcopy(self.irts[self.num_layers-1].approx.bases),
                    xs, 
                    neglogratios
                )

                sirt = TTSIRT(
                    updated_func,
                    bases=deepcopy(self.irts[self.num_layers-1].approx.bases),
                    approx=deepcopy(self.irts[self.num_layers-1].approx),
                    options=sirt_options,
                    input_data=deepcopy(input_data),
                    tt_data=deepcopy(self.irts[self.num_layers-1].approx.data),
                    tau=self.dirt_options.defensive
                )
        
        else:
            raise NotImplementedError()
        
        return sirt