import abc
from typing import Callable, Tuple

import torch

from .approx_bases import ApproxBases
from .input_data import InputData
from .options import TTOptions
from .tt_data import TTData


DEFAULT_TT_DATA = TTData()


class ApproxFunc(abc.ABC):

    def __init__(
        self, 
        func: Callable, 
        arg, # TODO: make a type annotation for this?
        options: TTOptions, 
        input_data: InputData
    ):

        self.options = options
        self.input_data = input_data
        self.data: TTData

        if isinstance(arg, ApproxFunc):
            self.bases = arg
            self.data = arg.data 
            self.options = arg.options

        else:
            if isinstance(arg, ApproxBases):
                self.bases = arg
            elif isinstance(arg, int):
                arg1 = DEFAULT_POLY
                arg2 = DEFAULT_DOMAIN
                d = arg
                self.bases = ApproxBases(arg1, arg2, d)
            else:
                raise Exception("Dimension should be a positive scalar.")
                
            self.data = DEFAULT_TT_DATA
        
        self.input_data.set_debug(func, self.bases)
        self.num_eval = 0
        self.errors = torch.zeros(self.bases.dim)
        self.l2_err = torch.inf
        self.linf_err = torch.inf
        return

    # TODO: add type annotations etc for the below abstractmethods.
    
    @abc.abstractmethod
    def eval_reference(
        self, 
        rs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the approximation to the target function for a set 
        of reference variables.
        """
        return
    
    @abc.abstractmethod 
    def grad_reference(
        self, 
        rs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the gradient of the approximation to the target 
        function for a set of reference variables.
        """
        return 
    
    @abc.abstractmethod
    def int_reference(self):
        """Integrates the approximation to the target function over the 
        reference domain (TODO: check this.).
        """
        return
    
    @property 
    def dim(self) -> torch.Tensor|int:
        return self.bases.dim

    def compute_relative_error(self) -> None:
        """TODO: write docstring."""

        if not self.input_data.is_debug:
            return
        
        approx = self.eval_reference(self.input_data.debug_z)
        self.l2_err, self.linf_err = self.input_data.relative_error(approx)
        return

    def eval(self, xs: torch.Tensor) -> torch.Tensor:
        """Evaluates the approximated function at a set of points in 
        the approximation domain.
        
        Parameters
        ----------
        xs:
            A matrix containing n sets of d-dimensional input 
            variables in the approximation domain. Each row contains a
            single input variable.
            
        Returns
        -------
        fxs:
            An n-dimensional vector containing the values of the 
            function at each x value.
        
        """

        rs = self.bases.domain2reference(xs)[0]
        fxs = self.eval_reference(rs)
        
        return fxs

    def grad(self, xs: torch.Tensor) -> torch.Tensor:
        """Evaluates the gradient of the approximation to the target 
        function at a set of points in the approximation domain.
        
        Parameters
        ----------
        xs: 
            A matrix containing n sets of d-dimensional input 
            variables in the approximation domain. Each row contains a
            single input variable.

        Returns
        -------
        gxs:
            TODO: finish this once grad_reference is done.

        """

        zs, dzdxs = self.bases.domain2reference(xs)
        gzs, fxs = self.grad_reference(self, zs)
        gxs = gzs * dzdxs
        return gxs, fxs