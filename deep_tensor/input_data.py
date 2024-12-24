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
        xs_samp: torch.Tensor|None=None, 
        xs_debug: torch.Tensor|None=None, 
        ps_debug: torch.Tensor|None=None
    ):
        
        if xs_samp is None:
            xs_samp = torch.tensor([])
        if xs_debug is None:
            xs_debug = torch.tensor([])
        if ps_debug is None:
            ps_debug = torch.tensor([])
        
        self.xs_samp = xs_samp
        self.xs_debug = xs_debug
        self.ps_debug = ps_debug

        self.ls_samp = torch.tensor([])
        self.ls_debug = torch.tensor([])
        self.count = 0

        return
        
    @property
    def is_debug(self) -> bool:
        """Flag that indicates whether debugging samples are available.
        """

        if self.ps_debug.numel() > 0 and self.ls_debug.numel() == 0:
            msg = "Debug samples must be transformed to the local domain."
            raise Exception(msg)
        
        flag = not self.ls_debug.numel() == 0
        return flag
    
    @property 
    def is_evaluated(self) -> bool:
        """Flag that indicates whether the approximation to the target 
        function has been evaluated for all samples.
        """
        return not self.ps_debug.numel() == 0

    def set_samples(
        self, 
        bases: ApproxBases, 
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
                self.ls_samp = bases.sample_measure_local(n)[0]
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
                self.ls_samp = bases.sample_measure_local(n)[0]
            else:
                self.ls_samp = bases.approx2local(self.xs_samp)[0]

        self.count = 0
        return
        
    def get_samples(self, n: int|None=None) -> torch.Tensor:
        """Returns a set of samples from the local domain.
        
        Parameters
        ----------
        n: 
            The number of samples to return.
        
        Returns
        -------
        ls:
            An n * d matrix containing samples from the local domain
            ([-1, 1]^d).
        
        """
        
        num_samples = self.ls_samp.shape[0]

        if self.count + n <= num_samples:
            indices = torch.arange(n) + self.count
            self.count += n
            return self.ls_samp[indices]

        n1 = num_samples - self.count + 1
        n2 = n - n1

        indices = torch.concatenate((
            torch.arange(self.count, num_samples), 
            torch.arange(n2)
        ))
        
        self.count = n2
        msg = "All samples have been used. Starting from the beginning."
        warnings.warn(msg)
        
        return self.ls_samp[indices]
        
    def set_debug(
        self, 
        target_func: Callable[[torch.Tensor], torch.Tensor], 
        bases: ApproxBases
    ) -> None:
        """Generates a set of samples to use to evaluate the quality of 
        the approximation to the target function.

        Parameters
        ----------
        target_func:
            A function that returns the value of the target function 
            for a given set of parameters from the local domain.
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
    
        self.ls_debug = bases.approx2local(self.xs_debug)[0]
        if self.ps_debug.numel() == 0:
            self.ps_debug = target_func(self.ls_debug)

        return
    
    def relative_error(
        self, 
        ps_approx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: write docstring."""
        
        # TODO: figure out if this is necessary (the function that 
        # calls this has an early return also)
        if not self.is_debug:
            return torch.inf, torch.inf 
        
        error_l2 = ((self.ps_debug - ps_approx).square().mean().sqrt()
                    / self.ps_debug.square().mean().sqrt())
        
        error_linf = ((self.ps_debug - ps_approx).abs().max()
                      / self.ps_debug.abs().max())

        return error_l2, error_linf
    
    def reset_counter(self) -> None:
        self.count = 0
        return