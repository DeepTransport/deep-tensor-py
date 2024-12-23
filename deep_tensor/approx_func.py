import abc
from typing import Callable, Tuple

import torch

from .approx_bases import ApproxBases
from .input_data import InputData
from .options import ApproxOptions
from .tt_data import TTData


class ApproxFunc(abc.ABC):

    def __init__(
        self, 
        func: Callable, 
        bases: ApproxBases,
        options: ApproxOptions,
        input_data: InputData,
        data: TTData|None=None
    ):

        self.bases = bases 
        self.dim = bases.dim
        self.options = options
        self.input_data = input_data
        self.data = TTData() if data is None else data  # TODO: change to approx_data?

        # if isinstance(arg, ApproxFunc):
        #     self.options = arg.options
        
        self.input_data.set_debug(func, self.bases)
        self.num_eval = 0
        self.errors = torch.zeros(self.bases.dim)
        self.l2_err = torch.inf
        self.linf_err = torch.inf
        return
    
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
        reference domain (TODO: check this).
        """
        return
    
    def compute_relative_error(self) -> None:
        """TODO: write docstring."""

        if not self.input_data.is_debug:
            return
        
        approx = self.eval_reference(self.input_data.ls_debug)
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

        rs = self.bases.approx2local(xs)[0]
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

        zs, dzdxs = self.bases.approx2local(xs)
        gzs, fxs = self.grad_reference(self, zs)
        gxs = gzs * dzdxs
        return gxs, fxs