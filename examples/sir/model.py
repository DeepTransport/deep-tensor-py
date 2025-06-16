import warnings

import numpy as np
from numpy import ndarray
import torch 
from torch import Tensor 
from scipy.integrate import solve_ivp


class SIRModel():

    def __init__(
        self, 
        S0: float = 99.0,
        I0: float = 1.0,
        R0: float = 0.0,
        t1: float = 5.0,
        t_eval: Tensor | ndarray | None = None
    ):
        """A simple SIR model. Setup replicated from Cui, Dolgov and 
        Zahm (2023).
        """
        if t_eval is None:
            t_eval = np.array([1.25, 2.5, 3.75, 5.0]) 
        self.S0 = S0 
        self.I0 = I0 
        self.R0 = R0
        self.y0 = np.array([S0, I0, R0])
        self.t_span = (0.0, t1)
        self.t_eval = np.array(t_eval)
        return
    
    @staticmethod 
    def sir_func(t, y: ndarray, b: float, g: float) -> ndarray:
        S, I, _ = y.reshape(3, -1)
        return np.array([-b*S*I, b*S*I - g*I, g*I]).flatten()
    
    def _solve_fwd(self, params: ndarray) -> Tensor:
        """Solves the forward problem with given parameters, and 
        returns the number of infected people at the desired times.
        """
        n_params = params.shape[0]
        sol = solve_ivp(
            fun=self.sir_func, 
            t_span=self.t_span,
            y0=self.y0.repeat(params.shape[0]), 
            args=params.T, 
            t_eval=self.t_eval
        )
        if not sol.success:
            msg = "Forward solver did not converge."
            warnings.warn(msg)
        return sol.y[n_params:(2*n_params)]  # infected only
    
    def solve_fwd(self, params: Tensor, n_batch: int = 10_000) -> Tensor:
        # For an unknown reason, the ODE solver fails with a 
        # segmentation fault if the dimension of the state is quite 
        # large.
        params = [params[i:i+n_batch] for i in range(0, params.shape[0], n_batch)]
        ys = np.vstack([self._solve_fwd(p) for p in params])
        return torch.tensor(ys)