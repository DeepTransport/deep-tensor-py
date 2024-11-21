from typing import Callable, Tuple
import warnings

import torch

from .approx_bases import ApproxBases
from .utils import info


class InputData():
    """A class containing sampling data used for initialising TT and 
    constructing various approximations.

    Properties
    ----------
    sample_x:
        Samples for initialising and enriching a TT approximation, 
        drawn from the approximation domain.
    sample_z: 
        The corresponding samples from the reference domain.
    debug_x:
        Samples used to estimate the approximation error, drawn from 
        the approximation domain.
    debug_z:
        The corresponding samples from the reference domain.
    debug_f: 
        Function evaluations at debug_z (??).

    """

    def __init__(
        self, 
        sample_x: torch.Tensor=torch.tensor([]), 
        debug_x: torch.Tensor=torch.tensor([]), 
        debug_f: torch.Tensor=torch.tensor([])
    ):
        
        self.sample_x = sample_x
        self.debug_x = debug_x
        self.debug_f = debug_f

        self.sample_z = torch.tensor([])
        self.debug_z = torch.tensor([])
        self.count = 0

        return
        
    @property
    def is_debug(self) -> bool:
        """Flag that indicates whether debugging samples are available.
        """

        if self.debug_z.numel() == 0 and self.debug_x.numel() > 0:
            msg = "Debug samples must be transformed to the reference domain."
            warnings.warn(msg)
        
        flag = not self.debug_z.numel() == 0
        return flag
    
    @property 
    def is_evaluated(self):
        """Flag that indicates whether the approximation to the target 
        function has been evaluated for all samples.
        """
        return not self.debug_f.numel() == 0

    def set_samples(
        self, 
        base: ApproxBases, 
        n: int|None=None
    ) -> torch.Tensor:
        """Generates the samples required for TT, given the 
        approximation basis.
        """
            
        if self.sample_x.numel() == 0:
            if n is not None:
                msg = ("Generating initialization samples from the " 
                       + "base measure.")
                info(msg)
                self.sample_z, _ = base.sample_measure_reference(n)
            else:
                msg = ("There are no initialization samples available. "
                       + "Please provide a sample set "
                       + "or specify a sample size.")
                raise Exception(msg)
        else:
            if n is not None and self.sample_x.shape[0] < n:
                # TODO: figure out what is going on in here -- do I 
                # need to adjust the x samples as well?
                msg = ("Not enough number of samples to initialise " 
                       + "functional tensor train.")
                warnings.warn(msg)
                self.sample_z, _ = base.sample_measure_reference(n)
            else:
                self.sample_z, _ = base.domain2reference(self.sample_x)

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
        :
            A set of samples of the requested size.
        
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
        
    def reset_counter(self) -> None:
        self.count = 0
        return
        
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
            target density.

        Returns
        -------
        None

        """

        if self.debug_x.numel() == 0:
            msg = ("Debug samples are not provided. " + 
                   "Turn off evaluation-based cross validation.")
            warnings.warn(msg)
            return
    
        self.debug_z, _ = bases.domain2reference(self.debug_x)
        if self.debug_f.numel() == 0:
            self.debug_f = func(self.debug_z)

        return
        
    def relative_error(
        self, 
        approx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # TODO: figure out if this is necessary (the function that 
        # calls this has an early return also)
        if not self.is_debug:
            return torch.inf, torch.inf 
        
        error_l2 = torch.mean((self.debug_f - approx) ** 2) ** 0.5 / torch.mean(self.debug_f ** 2) ** 0.5
        error_linf = torch.max(torch.abs(self.debug_f - approx)) / torch.max(torch.abs(self.debug_f))

        return error_l2, error_linf