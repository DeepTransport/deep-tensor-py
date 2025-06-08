from typing import Callable, Tuple
import warnings

import torch
from torch import Tensor

from .approx_bases import ApproxBases


class InputData():
    """Data used for building and evaluating the quality of a FTT.

    Parameters
    ----------
    xs_samp:
        A set of samples from the approximation domain, used to 
        construct the FTT approximation to the target function.
    xs_debug: 
        A set of samples from the approximation domain, used to 
        evaluate the quality of the FTT approximation to the target 
        function.
    fxs_debug:
        A vector containing the target function evaluated at each 
        sample in `xs_debug`.

    """

    def __init__(
        self, 
        xs_samp: Tensor | None = None, 
        xs_debug: Tensor | None = None, 
        fxs_debug: Tensor | None = None
    ):
        
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
        return not self.xs_debug.numel() == 0
    
    @property 
    def is_evaluated(self) -> bool:
        """Flag that indicates whether the approximation to the target 
        function has been evaluated for all debugging samples.
        """
        return not self.fxs_debug.numel() == 0

    def set_samples(self, bases: ApproxBases, n_samples: int) -> None:
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

        self.count = 0
        return
        
    def get_samples(self, n: int) -> Tensor:
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
        
        n_samples = self.ls_samp.shape[0]

        if self.count + n <= n_samples:
            indices = torch.arange(n) + self.count
            self.count += n
            return self.ls_samp[indices]

        n1 = n_samples - self.count + 1
        n2 = n - n1

        indices = torch.concatenate((torch.arange(self.count, n_samples), 
                                     torch.arange(n2)))
        self.count = n2
        msg = "All samples have been used. Starting from the beginning."
        warnings.warn(msg)
        return self.ls_samp[indices]
        
    def set_debug(
        self, 
        target_func: Callable[[Tensor], Tensor], 
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
        if not self.is_evaluated:
            self.fxs_debug = target_func(self.ls_debug)
        return
    
    def relative_error(
        self, 
        fxs_approx: Tensor
    ) -> Tuple[Tensor | float, Tensor | float]:
        """Estimates the L_2 and L_inf error between the target 
        function and FTT approximation using a set of samples.
        
        Parameters
        ----------
        fxs_approx:
            An n-dimensional vector containing the value of the target 
            function evaluated at each of the debugging samples.
        
        Returns
        -------
        error_l2:
            The estimate of the L2 error between the target function 
            and the FTT approximation.
        error_linf:
            The estimate of the L_inf error between the target function 
            and the FTT approximation.
        
        """
        
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