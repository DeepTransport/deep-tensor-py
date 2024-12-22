from typing import Callable, Tuple
import warnings

import torch

from .approx_bases import ApproxBases


class InputData():
    """A class containing sampling data used for building and 
    evaluating the quality of the approximation to a given target 
    function.
    """

    def __init__(
        self, 
        xs_samp: torch.Tensor=torch.tensor([]), 
        xs_debug: torch.Tensor=torch.tensor([]), 
        fxs_debug: torch.Tensor=torch.tensor([])
    ):
        
        self.xs_samp = xs_samp
        self.xs_debug = xs_debug
        self.fxs_debug = fxs_debug

        self.rs_samp = torch.tensor([])
        self.rs_debug = torch.tensor([])
        self.count = 0

        return
        
    @property
    def is_debug(self) -> bool:
        """Flag that indicates whether debugging samples are available.
        """

        if self.xs_debug.numel() > 0 and self.rs_debug.numel() == 0:
            msg = "Debug samples must be transformed to the reference domain."
            warnings.warn(msg)
        
        flag = not self.rs_debug.numel() == 0
        return flag
    
    @property 
    def is_evaluated(self) -> bool:
        """Flag that indicates whether the approximation to the target 
        function has been evaluated for all samples.
        """
        return not self.fxs_debug.numel() == 0

    def set_samples(
        self, 
        base: ApproxBases, 
        n: int|None=None
    ) -> torch.Tensor:
        """Generates the samples required for TT, given the 
        approximation basis.
        """
            
        if self.xs_samp.numel() == 0:
            if n is not None:
                msg = ("Generating initialization samples from the " 
                       + "base measure.")
                print(msg)
                self.rs_samp, _ = base.sample_measure_local(n)
            else:
                msg = ("There are no initialization samples available. "
                       + "Please provide a sample set "
                       + "or specify a sample size.")
                raise Exception(msg)
        else:
            if n is not None and self.xs_samp.shape[0] < n:
                # TODO: figure out what is going on in here -- do I 
                # need to adjust the x samples as well?
                msg = ("Not enough number of samples to initialise " 
                       + "functional tensor train.")
                warnings.warn(msg)
                self.sample_z, _ = base.sample_measure_local(n)
            else:
                self.sample_z, _ = base.approx2local(self.xs_samp)

        self.count = 0
        return
        
    def get_samples(self, n: int|None=None) -> torch.Tensor:
        """Returns a set of samples from the reference domain.
        
        Parameters
        ----------
        n: 
            Number of samples to return.
        
        Returns
        -------
        rs:
            An n * d matrix containing the requested samples.
        
        """
        
        num_samples = self.sample_z.shape[0]

        if self.count + n <= num_samples:
            indices = torch.arange(n) + self.count
            self.count += n
            return self.sample_z[indices]

        n1 = num_samples - self.count + 1
        n2 = n - n1

        indices = torch.concatenate((
            torch.arange(self.count, num_samples), 
            torch.arange(n2)
        ))
        
        self.count = n2
        msg = "All samples have been used. Starting from the beginning."
        warnings.warn(msg)
        
        return self.sample_z[indices]
        
    def set_debug(
        self, 
        func: Callable, 
        bases: ApproxBases
    ) -> None:
        """Generates a set of samples to use to evaluate the quality of 
        the approximation to the target density.

        Parameters
        ----------
        func:
            Function that returns the value of the target function for 
            a given set of parameters from the reference domain.
        bases:
            The set of bases used to construct the approximation to the
            target function.

        Returns
        -------
        None

        """

        if self.xs_debug.numel() == 0:
            msg = ("Debug samples are not provided. " + 
                   "Turn off evaluation-based cross validation.")
            warnings.warn(msg)
            return
    
        self.rs_debug, _ = bases.approx2local(self.xs_debug)
        if self.fxs_debug.numel() == 0:
            self.fxs_debug = func(self.rs_debug)

        return
    
    def relative_error(
        self, 
        approx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: write docstring."""
        
        # TODO: figure out if this is necessary (the function that 
        # calls this has an early return also)
        if not self.is_debug:
            return torch.inf, torch.inf 
        
        error_l2 = ((self.fxs_debug - approx).square().mean().sqrt()
                    / self.fxs_debug.square().mean().sqrt())
        
        error_linf = ((self.fxs_debug - approx).abs().max()
                      / self.fxs_debug.abs().max())

        return error_l2, error_linf
    
    def reset_counter(self) -> None:
        self.count = 0
        return