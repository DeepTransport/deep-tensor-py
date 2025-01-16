from typing import Callable, Tuple
import warnings

import torch

from .approx_bases import ApproxBases


class InputData():

    def __init__(
        self, 
        xs_samp: torch.Tensor|None=None, 
        xs_debug: torch.Tensor|None=None, 
        fxs_debug: torch.Tensor|None=None
    ):
        """A class containing sampling data used for building and 
        evaluating the quality of the approximation to a given target 
        function.

        Parameters
        ----------
        xs_samp:
            A set of samples from the approximation domain, used to 
            construct the FTT approximation to the target density 
            function.
        xs_debug: 
            A set of samples from the approximation domain, used to 
            evaluate the quality of the FTT approximation the the 
            target density function.
        fxs_debug:
            A vector containing the approximation to the target density 
            function evaluated at each sample in xs_debug.

        """

        if (xs_debug.numel() == 0) ^ (fxs_debug.numel() == 0):
            msg = ("If debugging samples are provided, the value of "
                   + "the target function at each sample must also "
                   + "be specified (and vice versa).")
            raise Exception(msg)
        
        if xs_samp is None:
            xs_samp = torch.tensor([])
        if xs_debug is None:
            xs_debug = torch.tensor([])
        if fxs_debug is None:
            fxs_debug = torch.tensor([])
        
        self.xs_samp = xs_samp
        self.xs_debug = xs_debug
        self.fxs_debug = fxs_debug

        self.ls_samp = torch.tensor([])
        self.ls_debug = torch.tensor([])
        
        self.count = 0
        return
        
    @property
    def is_debug(self) -> bool:
        """Flag that indicates whether debugging samples are available.
        """

        # if self.xs_debug.numel() > 0 and self.ls_debug.numel() == 0:
        #     msg = "Debug samples must be transformed to the local domain."
        #     raise Exception(msg)
        
        flag = not self.xs_debug.numel() == 0
        return flag
    
    @property 
    def is_evaluated(self) -> bool:
        """Flag that indicates whether the approximation to the target 
        function has been evaluated for all samples.
        """
        return not self.fxs_debug.numel() == 0

    def set_samples(
        self, 
        bases: ApproxBases, 
        n_samples: int
    ) -> torch.Tensor:
        """Generates the samples used to construct the FTT (if not 
        specified during initialisation), then transforms these samples
        to the local domain.

        Parameters
        ----------
        bases: 
            The set of bases used to construct the approximation.
        n_samples:
            The number of samples to generate.

        Returns
        -------
        None

        Notes
        -----
        Updates self.ls_samp.

        """

        if self.xs_samp.numel() == 0:
            msg = ("Generating initialization samples from the " 
                    + "base measure.")
            print(msg)
            self.ls_samp = bases.sample_measure_local(n_samples)[0]        
        else:
            if self.xs_samp.shape[0] < n_samples:
                msg = ("Not enough number of samples to initialise " 
                        + "functional tensor train.")
                raise Exception(msg)
            self.ls_samp = bases.approx2local(self.xs_samp)[0]

        return
        
    def get_samples(self, n: int|None=None) -> torch.Tensor:
        """Returns a set of samples from the local domain.
        
        Parameters
        ----------
        n: 
            The number of samples to return.
        
        Returns
        -------
        ls_samp:
            An n * d matrix containing samples from the local domain.
        
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
        self.ls_debug = bases.approx2local(self.xs_debug)[0]
        if self.fxs_debug.numel() == 0:
            self.fxs_debug = target_func(self.ls_debug)
        return
    
    def relative_error(
        self, 
        fxs_approx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: write docstring."""
        
        if not self.is_debug:
            return torch.inf, torch.inf 
        
        error_l2 = ((self.fxs_debug - fxs_approx).square().mean().sqrt()
                    / self.fxs_debug.square().mean().sqrt())
        
        error_linf = ((self.fxs_debug - fxs_approx).abs().max()
                      / self.fxs_debug.abs().max())

        return error_l2, error_linf
    
    def reset_counter(self) -> None:
        self.count = 0
        return